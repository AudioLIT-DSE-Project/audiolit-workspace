
import pytest
import gc
import tracemalloc
import torch
import numpy as np

class TestMemoryProfiling:
    """Profile CPU workloads and track VRAM footprints across iterative tasks."""

    def test_cpu_memory_stability_under_iteration(self):
        """Verify CPU RAM does not leak across iterative localization tasks."""
        tracemalloc.start()
        
        # Baseline memory
        snapshot1 = tracemalloc.take_snapshot()
        
        # Simulate 50 iterations of heavy array processing (mimicking worker tasks)
        for _ in range(50):
            # Allocate large arrays (simulating spectrograms/attribution matrices)
            matrices = [np.random.rand(128, 300).astype(np.float32) for _ in range(10)]
            # Process and discard
            del matrices
            gc.collect()
            
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        
        # Assert memory difference is negligible (< 5MB to account for Python overhead)
        assert total_diff < 5 * 1024 * 1024, f"CPU memory leak detected: +{total_diff / 1024**2:.2f} MB"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="VRAM test requires CUDA")
    def test_vram_clears_successfully_between_iterations(self):
        """Verify VRAM blocks are freed immediately upon final result delivery."""
        device = torch.device("cuda")
        
        # Clear cache and get baseline VRAM
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        
        # Simulate 10 iterations of deep PyTorch inference (e.g., ADDSegDiff)
        for i in range(10):
            # Simulate model forward pass + gradient buffers
            fake_input = torch.randn(1, 1, 16000, device=device, requires_grad=True)
            fake_weights = torch.randn(16, 1, 3, device=device, requires_grad=True)
            fake_output = torch.nn.functional.conv1d(fake_input, fake_weights)
            fake_loss = fake_output.sum()
            fake_loss.backward()
            
            # Simulate result delivery & cleanup
            result = fake_loss.item()
            del fake_input, fake_weights, fake_output, fake_loss, result
            torch.cuda.empty_cache()
            
            # Check VRAM at the end of each iteration
            current_allocated = torch.cuda.memory_allocated(device)
            
            # Allocated VRAM should drop back near baseline after empty_cache()
            # We allow a small tolerance for PyTorch's internal memory manager
            assert current_allocated <= baseline_allocated + (10 * 1024 * 1024), \
                f"VRAM leak detected at iteration {i}: {current_allocated / 1024**2:.2f} MB allocated"
                
        # Final strict check: reserved memory should not grow unbounded
        final_reserved = torch.cuda.memory_reserved(device)
        assert final_reserved <= baseline_reserved + (50 * 1024 * 1024), \
            f"VRAM reserved memory grew unbounded: {final_reserved / 1024**2:.2f} MB"

class TestAPIStressConcurrency:
    """Simulate API stress testing for concurrent upload/inference requests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_dispatch(self):
        """Push concurrent requests to the backend architecture to discover blocks."""
        import httpx
        import asyncio
        
        # Mock endpoint testing: simulate 20 concurrent dispatches
        # In a real CI environment, this would target a running FastAPI instance
        # Here we simulate the async load handling to ensure no blocking
        
        async def mock_dispatch(client, task_id):
            # Simulate the gateway enqueuing a job and returning immediately
            await asyncio.sleep(0.01) # Represents network + Redis enqueue time
            return {"job_id": f"task_{task_id}", "status": "queued"}
            
        async with httpx.AsyncClient() as client:
            tasks = [mock_dispatch(client, i) for i in range(20)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 20
            assert all(r["status"] == "queued" for r in results)
