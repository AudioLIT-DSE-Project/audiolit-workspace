"""Unit tests for SHA-256 Audio Payload Hashing Middleware (SRS FR4)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.core.redis import generate_audio_hash_from_bytes, get_audio_hash_dependency

class TestAudioHashing:
    """Verify deterministic SHA-256 checksum generation."""

    def test_identical_files_yield_same_hash(self):
        """DoD: Uploading identical files yields the exact same SHA-256 string."""
        # Simulate identical raw audio byte streams
        audio_bytes_1 = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00" * 1000
        audio_bytes_2 = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00" * 1000
        
        hash1 = generate_audio_hash_from_bytes(audio_bytes_1)
        hash2 = generate_audio_hash_from_bytes(audio_bytes_2)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_minor_variations_generate_different_hash(self):
        """DoD: Minor audio variations generate a completely different hash."""
        audio_bytes_1 = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00" * 1000
        # Change the very last byte slightly
        audio_bytes_2 = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00\x20\x00\x00\x01" * 1000
        
        hash1 = generate_audio_hash_from_bytes(audio_bytes_1)
        hash2 = generate_audio_hash_from_bytes(audio_bytes_2)
        
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_fastapi_dependency_intercepts_and_hashes(self):
        """Verify the dependency interceptor reads chunks and returns the correct hash."""
        audio_bytes = b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00" * 5000
        
        # Mock the FastAPI UploadFile object
        mock_upload_file = MagicMock()
        
        # Simulate reading in 8KB chunks
        chunk_size = 8192
        chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]
        chunks.append(b"") # Signal end of file
        
        read_call_count = 0
        async def mock_read(size):
            nonlocal read_call_count
            if read_call_count < len(chunks):
                val = chunks[read_call_count]
                read_call_count += 1
                return val
            return b""
            
        async def mock_seek(pos):
            pass
            
        mock_upload_file.read = mock_read
        mock_upload_file.seek = mock_seek
        mock_upload_file.filename = "test_audio.wav"
        
        # Execute the dependency
        audio_hash = await get_audio_hash_dependency(audio_file=mock_upload_file)
        
        # Verify the hash matches the raw bytes calculation
        expected_hash = generate_audio_hash_from_bytes(audio_bytes)
        assert audio_hash == expected_hash
        
        # Verify it actually read the file in chunks (did not load all at once)
        assert read_call_count > 1
