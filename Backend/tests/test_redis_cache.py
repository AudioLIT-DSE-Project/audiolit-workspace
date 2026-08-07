"""Unit tests for Redis Tensor Serialization & Cache-by-Hash (SRS FR4)."""
import pytest
import time
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from app.core.redis import cache_manager, cached_inference

class TestRedisCacheManager:
    """Test SHA-256 hashing and msgpack/NumPy tensor serialization."""

    def test_sha256_key_generation_is_deterministic(self):
        """Verify FR4.1: Key scheme is unique and deterministic."""
        key1 = cache_manager._generate_key("audio_123.wav", "whisper-base", "asr", {"temperature": 0.5})
        key2 = cache_manager._generate_key("audio_123.wav", "whisper-base", "asr", {"temperature": 0.5})
        key3 = cache_manager._generate_key("audio_123.wav", "whisper-base", "asr", {"temperature": 0.6})
        
        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("audiolit:tensor:")
        assert len(key1.split(":")[-1]) == 64  # SHA-256 hex length

    def test_numpy_serialization_roundtrip(self):
        """Verify multi-dimensional arrays are compressed and restored correctly."""
        original_data = {
            "attribution": np.random.rand(128, 300).astype(np.float32),
            "logits": np.array([0.1, 0.9], dtype=np.float64)
        }
        
        # Mock the actual Redis client set/get to test serialization logic
        with patch.object(cache_manager, 'client') as mock_client:
            mock_get_storage = {}
            
            def mock_set(key, value, ttl):
                mock_get_storage[key] = value
            def mock_get(key):
                return mock_get_storage.get(key)
                
            mock_client.setex.side_effect = mock_set
            mock_client.get.side_effect = mock_get
            
            cache_manager.set("test_key", original_data)
            retrieved_data = cache_manager.get("test_key")
            
            assert retrieved_data is not None
            assert np.array_equal(retrieved_data["attribution"], original_data["attribution"])
            assert np.array_equal(retrieved_data["logits"], original_data["logits"])


class TestCachedInferenceInterceptor:
    """Verify the decorator intercepts and bypasses heavy execution paths."""

    def test_sub_10ms_cache_hit_bypasses_execution(self, caplog):
        """Verify DoD: Duplicate query reads from Redis in <10ms, avoiding model forward pass."""
        
        @cached_inference(audio_arg="audio_path", model_arg="model_id", task_name="mock_inference")
        def mock_heavy_inference(audio_path: str, model_id: str, params: dict):
            # Simulate a 500ms heavy PyTorch forward/backward pass
            time.sleep(0.5)
            return {"attribution": np.random.rand(128, 300).astype(np.float32)}
            
        audio_path = "test_audio.wav"
        model_id = "wav2vec2"
        params = {"layer": 5}
        
        # 1. First call: Cache Miss -> should execute heavy function
        with patch.object(cache_manager, 'client') as mock_client:
            mock_get_storage = {}
            def mock_set(key, value, ttl):
                mock_get_storage[key] = value
            def mock_get(key):
                return mock_get_storage.get(key)
                
            mock_client.setex.side_effect = mock_set
            mock_client.get.side_effect = mock_get
            
            with caplog.at_level(logging.INFO):
                start_time_miss = time.time()
                result_miss = mock_heavy_inference(audio_path=audio_path, model_id=model_id, params=params)
                duration_miss = (time.time() - start_time_miss) * 1000
                
                assert duration_miss > 500  # Heavy execution ran
                assert "CACHE MISS: Executing PyTorch inference" in caplog.text
                
            # 2. Second call: Cache Hit -> should bypass heavy function entirely
            with caplog.at_level(logging.INFO):
                start_time_hit = time.time()
                result_hit = mock_heavy_inference(audio_path=audio_path, model_id=model_id, params=params)
                duration_hit = (time.time() - start_time_hit) * 1000
                
                # DoD: Must read in under 10ms
                assert duration_hit < 10.0, f"Cache hit took {duration_hit:.2f}ms, expected <10ms"
                assert "CACHE HIT:" in caplog.text
                assert "Bypassing PyTorch execution graph completely" in caplog.text
                
            # Verify data integrity
            assert np.array_equal(result_miss["attribution"], result_hit["attribution"])
