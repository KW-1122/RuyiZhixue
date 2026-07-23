import tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ruyitutor.storage import Storage

class ConcurrencyTests(unittest.TestCase):
    def test_parallel_sqlite_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Storage(Path(folder)/"test.db")
            def write(i): store.save_message("c",f"s{i%3}","user",f"q{i}")
            with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(write,range(40)))
            total=sum(len(store.history(f"s{i}")) for i in range(3))
            self.assertEqual(total,40)
