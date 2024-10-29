import numpy as np
import os
import pytest
import shutil

from pathlib import Path
from time import sleep

from benchmark.utils.cache.lock import DirLock
from benchmark.utils.cache.disk_cache import HDF5DiskCache


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for testing.

    """
    dirpath = Path(f"cachetests_{np.random.randint(1e6)}")
    yield dirpath
    if dirpath.exists():
        shutil.rmtree(dirpath)


def test_acquire_and_release_lock(temp_dir):
    """
    Test that a lock can be acquired and released.

    """
    lock = DirLock(temp_dir)
    lock.acquire()
    assert Path(temp_dir).exists()
    lock.release()
    assert not Path(temp_dir).exists()


def test_lock_age(temp_dir):
    """
    Test that the age of a lock can be retrieved.

    """
    lock = DirLock(temp_dir)
    lock.acquire()
    sleep(2)
    age = lock.age()
    assert age is not None
    assert age >= 0
    lock.release()


def test_context_manager(temp_dir):
    """
    Test that a lock can be acquired and released using a context manager.

    """
    with DirLock(temp_dir) as lock:
        assert Path(temp_dir).exists()
    assert not Path(temp_dir).exists()


def test_lock_blocks_others(temp_dir):
    lock1 = DirLock(temp_dir, retry_interval=0.01, timeout=2)
    lock1.acquire()

    with pytest.raises(TimeoutError):
        lock2 = DirLock(temp_dir, retry_interval=0.01, timeout=2)
        lock2.acquire()


def test_lock_age_when_not_acquired(temp_dir):
    """
    Test that the age of a lock is None when it has not been acquired.

    """
    lock = DirLock(temp_dir)
    age = lock.age()
    assert age is None


@pytest.fixture
def cache(temp_dir):
    """
    Initialize an HDF5DiskCache object for testing.

    """
    return HDF5DiskCache(temp_dir)


def test_set_and_get_item(cache):
    """
    Test that an item can be set and retrieved from the cache.

    """
    cache.set("key1", np.array([1, 2, 3]))
    result = cache.get("key1")
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_set_and_get_item_special_syntax(cache):
    """
    Test that an item can be set and retrieved from the cache using special syntax.

    """
    cache["key1"] = np.array([1, 2, 3])
    result = cache["key1"]
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_update_item(cache):
    """
    Test that an item can be updated in the cache.

    """
    cache["key1"] = np.array([1, 2, 3])
    cache["key1"] = np.array([4, 5, 6])
    np.testing.assert_array_equal(cache["key1"], np.array([4, 5, 6]))


def test_get_nonexistent_key(cache):
    """
    Test that a default value is returned when a nonexistent key is requested.

    """
    result = cache.get("nonexistent", default="default_value")
    assert result == "default_value"


def test_clear_cache(cache):
    """
    Test that the cache can be cleared.

    """
    cache.set("key1", np.array([1, 2, 3]))
    cache.set("key2", np.array([4, 5, 6]))
    cache.clear()
    assert len(cache) == 0


def test_cache_contains(cache):
    """
    Test that the cache can be checked for key existence.

    """
    cache.set("key1", np.array([1, 2, 3]))
    assert "key1" in cache
    assert "nonexistent" not in cache


def test_cache_length(cache):
    """
    Test that the length of the cache can be retrieved.

    """
    cache.set("key1", np.array([1, 2, 3]))
    cache.set("key2", np.array([4, 5, 6]))
    assert len(cache) == 2


def test_cache_iteration(cache):
    """
    Test that the cache can be iterated over.

    """
    keys = ["key1", "key2", "key3"]
    for key in keys:
        cache.set(key, np.array([1, 2, 3]))
    iterated_keys = list(cache)
    assert set(iterated_keys) == set(keys)


def test_clear_stale_locks(temp_dir):
    """
    Test that stale locks can be cleared.

    """
    # Assuming _clear_stale_locks is public or can be called directly
    cache = HDF5DiskCache(temp_dir, lock_timeout=0)
    cache.locks[0].acquire()  # Acquire a lock
    sleep(2)

    # Assert that the lock is acquired
    assert cache.locks[0].lock_dir.exists()

    # Since the lock has a timeout of 0, it should be stale
    cache._clear_stale_locks()

    # Assert that the lock is cleared
    assert all(not lock.lock_dir.exists() for lock in cache.locks.values())

    # Now do it again but with a timeout of 999 seconds and assert that the lock is not cleared
    cache = HDF5DiskCache(temp_dir, lock_timeout=999)
    cache.locks[0].acquire()  # Acquire a lock

    # Assert that the lock is acquired
    assert cache.locks[0].lock_dir.exists()

    # Since the lock has a timeout of 999 seconds, it should not be stale
    cache._clear_stale_locks()

    # Assert that the lock is not cleared
    assert cache.locks[0].lock_dir.exists()

    # Clear the lock
    cache.locks[0].release()


def test_reading_scalar(cache):
    """
    Test that a scalar can be read from the cache and that it
    is returned as a scalar and not as a numpy array.

    """
    cache["scalar"] = 5
    assert cache["scalar"] == 5


def test_extra_info(cache):
    """
    Test that we can save and retrieve the extra information dictionary together with the array.
    """
    cache["key1"] = (np.array([1, 2, 3]), {"int": 5, "float": 0.5, "str": "five"})
    _, extra_info = cache["key1"]
    assert extra_info == {"int": 5, "float": 0.5, "str": "five"}


def test_update_extra_info(cache):
    """
    Test that the extra info gets correctly updated
    """
    cache["key1"] = (np.array([1, 2, 3]), {"int": 5, "float": 0.5, "str": "five"})
    cache["key1"] = (
        np.array([1, 2, 3]),
        {"newint": 6, "newfloat": 0.6, "newstr": "six"},
    )
    _, extra_info = cache["key1"]
    assert extra_info == {"newint": 6, "newfloat": 0.6, "newstr": "six"}
