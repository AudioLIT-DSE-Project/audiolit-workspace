"""Unit tests for Deterministic Cache-by-Hash Architecture (SRS FR4)."""
import pytest
import time
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from app.core.redis import cache_manager, cached_inference

class TestRedisCacheManager:
    """Test SHA-256 hashing, serialization, and corruption handling."""

    def test_deterministic_key_generation(self):
        """Verify FR4.1: Key scheme is unique and deterministic."""
        audio_bytes = b"fake_audio_data"
        model_id = "whisper-base"
        task = "asr"
        params = {"temperature": 0.5}
        
        key1 = cache_manager._generate_key(audio_bytes, model_id, task, params)
        key2 = cache_manager._generate_key(audio_bytes, model_id, task, params)
        
        assert key1 == key2
        assert key1.startswith("audiolit:tensor:")
        assert len(key1.split(":")[-1]) == 64  # SHA-256 hex length

    def test_key_sensitivity_to_variations(self):
        """Verify minor audio or param variations generate completely different hashes."""
        audio_bytes = b"fake_audio_data"
        model_id = "whisper-base"
        task = "asr"
        
        key1 = cache_manager._generate_key(audio_bytes, model_id, task, {"temp": 0.5})
        key2 = cache_manager._generate_key(audio_bytes, model_id, task, {"temp": 0.6})
        key3 = cache_manager._generate_key(b"different_audio", model_id, task, {"temp": 0.5})
        key4 = cache_manager._generate_key(audio_bytes, "wav2vec2", task, {"temp": 0.5})
        
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_numpy_serialization_roundtrip(self):
        """Verify multi-dimensional arrays are compressed and restored correctly."""
        original_data = {
            "attribution": np.random.rand(128, 300).astype(np.float32),
            "logits": np.array([0.1, 0.9], dtype=np.float64)
        }
        
        with patch.object(cache_manager, 'client') as mock_client:
            mock_get_storage = {}
            def mock_set(key, value, ex=None):
                mock_get_storage[key] = value
            def mock_get(key):
                return mock_get_storage.get(key)
                
            mock_client.set.side_effect = mock_set
            mock_client.get.side_effect = mock_get
            
            cache_manager.set("test_key", original_data)
            retrieved_data = cache_manager.get("test_key")
            
            assert retrieved_data is not None
            assert np.array_equal(retrieved_data["attribution"], original_data["attribution"])

    def test_lz4_compression_for_large_tensors(self):
        """Verify lz4-compress values above ~1 MB."""
        large_array = np.random.rand(300, 1000).astype(np.float32)  # ~1.2MB
        original_data = {"tensor": large_array}
        
        with patch.object(cache_manager, 'client') as mock_client:
            mock_get_storage = {}
            def mock_set(key, value, ex=None):
                mock_get_storage[key] = value
            def mock_get(key):
                return mock_get_storage.get(key)
                
            mock_client.set.side_effect = mock_set
            mock_client.get.side_effect = mock_get
            
            cache_manager.set("large_key", original_data)
            
            # Verify it was compressed
            stored_data = mock_get_storage["large_key"]
            assert stored_data.startswith(b'LZ4:')
            
            # Verify it decompresses correctly
            retrieved_data = cache_manager.get("large_key")
            assert np.array_equal(retrieved_data["tensor"], original_data["tensor"])

    def test_corrupt_cache_deletes_and_recomputes(self, caplog):
        """Verify a corrupt cached value is detected, discarded, and recomputed."""
        with patch.object(cache_manager, 'client') as mock_client:
            mock_client.get.return_value = b"corrupt_data_not_valid_msgpack"
            mock_client.delete = MagicMock()
            
            with caplog.at_level(logging.WARNING):
                result = cache_manager.get("corrupt_key")
                
            assert result is None
            mock_client.delete.assert_called_once_with("corrupt_key")
            assert "Corrupt cache entry detected" in caplog.text


class TestCachedInferenceInterceptor:
    """Verify the decorator intercepts and bypasses heavy execution paths."""

    def test_hit_bypass_miss_enqueue(self, caplog):
        """Verify DoD: Duplicate query reads from Redis, avoiding model forward pass."""
        
        @cached_inference(audio_bytes_arg="audio_bytes", model_arg="model_id", task_name="mock_inference")
        def mock_heavy_inference(audio_bytes: bytes, model_id: str, params: dict):
            time.sleep(0.5)  # Simulate heavy PyTorch execution
            return {"attribution": np.random.rand(128, 300).astype(np.float32)}
            
        audio_bytes = b"test_audio_bytes"
        model_id = "wav2vec2"
        params = {"layer": 5}
        
        with patch.object(cache_manager, 'client') as mock_client:
            mock_get_storage = {}
            def mock_set(key, value, ex=None, nx=False):
                if nx and key in mock_get_storage:
                    return False
                mock_get_storage[key] = value
                return True
            def mock_get(key):
                return mock_get_storage.get(key)
            def mock_delete(key):
                mock_get_storage.pop(key, None)
                
            mock_client.set.side_effect = mock_set
            mock_client.get.side_effect = mock_get
            mock_client.delete.side_effect = mock_delete
            
            with caplog.at_level(logging.INFO):
                # 1. First call: Cache Miss -> should execute heavy function
                start_time_miss = time.time()
                result_miss = mock_heavy_inference(audio_bytes=audio_bytes, model_id=model_id, params=params)
                duration_miss = (time.time() - start_time_miss) * 1000
                
                assert duration_miss > 500
                assert "CACHE MISS: Executing PyTorch inference" in caplog.text
                
                # 2. Second call: Cache Hit -> should bypass heavy function entirely
                start_time_hit = time.time()
                result_hit = mock_heavy_inference(audio_bytes=audio_bytes, model_id=model_id, params=params)
                duration_hit = (time.time() - start_time_hit) * 1000
                
                # DoD: Must read in under 200ms (actually will be < 10ms)
                assert duration_hit < 200.0, f"Cache hit took {duration_hit:.2f}ms, expected <200ms"
                assert "CACHE HIT:" in caplog.text
                assert "Bypassing PyTorch execution graph completely" in caplog.text
                
            # Verify data integrity (byte-identical result)
            assert np.array_equal(result_miss["attribution"], result_hit["attribution"])
