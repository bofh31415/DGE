import unittest
import os
import shutil
import zipfile
import glob

# We will implement these in run_tinystories_gsm8k_chain.py or a helper
# For imports, we might need the sys.path hack again
import sys
sys.path.append(os.getcwd())
try:
    from run_tinystories_gsm8k_chain import create_chunked_archive, restore_chunked_archive
except ImportError:
    # If the functions don't exist yet, we can def them here for the test structure 
    # or just assume we will implement them.
    # We'll define stubs in the test setup if import fails, but TDD prefers failing import.
    pass

class TestArchiving(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_archiving_data"
        self.restore_dir = "test_archiving_restore"
        self.output_dir = "test_archiving_output"
        
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.restore_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a large-ish dummy file (e.g. 10MB) and some small files
        with open(os.path.join(self.test_dir, "large_file.bin"), "wb") as f:
            f.write(os.urandom(10 * 1024 * 1024)) # 10MB
            
        with open(os.path.join(self.test_dir, "config.json"), "w") as f:
            f.write('{"test": true}')

    def tearDown(self):
        if os.path.exists(self.test_dir): shutil.rmtree(self.test_dir)
        if os.path.exists(self.restore_dir): shutil.rmtree(self.restore_dir)
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)

    def test_chunking_and_restoration(self):
        # Import inside test to allow implementation after test creation
        from run_tinystories_gsm8k_chain import create_chunked_archive, restore_chunked_archive
        
        # ACT 1: Create Archive with small chunk size (e.g., 2MB) to force splitting
        # 10MB file should produce ~5 chunks
        archive_name = os.path.join(self.output_dir, "test_archive")
        chunks = create_chunked_archive(self.test_dir, archive_name, chunk_size_mb=2)
        
        # ASSERT 1: Files created
        chunk_files = glob.glob(archive_name + ".*")
        self.assertTrue(len(chunk_files) >= 5, f"Expected >= 5 chunks, got {len(chunk_files)}")
        self.assertTrue(all(os.path.getsize(f) <= 2.1 * 1024 * 1024 for f in chunk_files), "Chunks should be roughly chunk_size")
        
        # ACT 2: Restore
        restore_chunked_archive(archive_name, self.restore_dir)
        
        # ASSERT 2: Integrity
        orig_large = os.path.join(self.test_dir, "large_file.bin")
        restored_large = os.path.join(self.restore_dir, "large_file.bin")
        
        self.assertTrue(os.path.exists(restored_large))
        self.assertTrue(os.path.exists(os.path.join(self.restore_dir, "config.json")))
        
        with open(orig_large, "rb") as f1, open(restored_large, "rb") as f2:
            self.assertEqual(f1.read(), f2.read(), "Restored binary content mismatch")

if __name__ == "__main__":
    unittest.main()
