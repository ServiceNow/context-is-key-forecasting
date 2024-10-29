import hashlib
from pathlib import Path
import h5py
import numpy as np
import logging

from .lock import DirLock


logger = logging.getLogger("HDF5DiskCache")


class HDF5DiskCache:
    """
    A simple disk-based cache using HDF5 files to store key-value pairs. The cache is divided into
    multiple bins based on the hash of the key to distribute the keys across multiple files. This
    is done to avoid creating many small files and to improve performance when doing parallel reads
    and writes.

    Parameters:
    -----------
    cache_dir : str
        The directory to store the cache files.
    lock_timeout : int, default 60
        The timeout in seconds before a lock is considered stale.
    num_bins : int, default 10
        The number of bins to distribute keys across.
    """

    def __init__(self, cache_dir, lock_timeout=60, num_bins=10):
        self.cache_dir = Path(cache_dir)
        self.lock_timeout = (
            lock_timeout  # Timeout in seconds before a lock is considered stale
        )
        self.num_bins = num_bins  # Number of bins to distribute keys
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate instances for all bins
        self.locks = {
            bin_number: DirLock(
                lock_dir=self._get_lock_dir(bin_number), timeout=self.lock_timeout
            )
            for bin_number in range(self.num_bins)
        }
        self._clear_stale_locks()

        logger.debug(
            f"Initialized HDF5DiskCache with directory: {self.cache_dir}, "
            f"lock_timeout: {self.lock_timeout}, num_bins: {self.num_bins}"
        )

    def _clear_stale_locks(self):
        """
        Delete all locks that are older than the lock timeout to prevent stale locks from
        preventing access to the cache (e.g., if a process crashes while holding a lock).

        Caveats:
        --------
        * This can lead to a race condition if another process runs this function and then
          acquires the lock at an intermediate state of this method. This would cause the
          current process to believe a lock is stale and then release the new lock of the
          other process. This is why we don't include this as part of the lock acquisition
          mechanism. Since it's done only once at cache initialization, the likelihood of
          this happening is low. We will know if this happens because HDF5 will raise an
          exception. In that case, just rerun the code.

        """
        logger.debug("Clearing stale locks")
        for lock in self.locks.values():
            age_seconds = lock.age()
            if age_seconds is not None and age_seconds > self.lock_timeout:
                logger.debug(f"Deleting stale lock at {lock.lock_dir}")
                lock.release()

    def _get_bin_id(self, key):
        """Get the bin number associated with the given key."""
        bin_id = int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_bins
        logger.debug(f"Computed bin_id: {bin_id} for key: {key}")
        return bin_id

    def _get_bin_path(self, bin_id):
        """Get the file path for the bin associated with the given bin_id."""
        bin_path = self.cache_dir / f"bin_{bin_id}.h5"
        logger.debug(f"Computed bin_path: {bin_path} for bin_id: {bin_id}")
        return bin_path

    def _get_lock_dir(self, bin_id):
        """Get the directory path for the lock associated with a given bin."""
        lock_dir = self.cache_dir / f"lock_{bin_id}"
        logger.debug(f"Computed lock_dir: {lock_dir} for bin_id: {bin_id}")
        return lock_dir

    def __getitem__(self, key):
        """Retrieve a value from the cache."""
        bin_id = self._get_bin_id(key)
        bin_path = self._get_bin_path(bin_id)
        lock = self.locks[bin_id]

        logger.debug(f"Attempting to retrieve key: {key} from bin_id: {bin_id}")
        with lock:  # Acquire lock before reading
            if not bin_path.exists():
                logger.debug(
                    f"Bin file for bin_id: {bin_id} does not exist, so key does not exist."
                )
                raise KeyError(f"Key '{key}' not found in cache.")
            with h5py.File(bin_path, "r") as f:
                if key in f:
                    ds = f[key]
                    value = ds[()]
                    attributes = {}
                    attributes.update(ds.attrs)
                    if len(attributes) > 0:
                        logger.debug(
                            f"Successfully retrieved key: {key} from bin_id: {bin_id}, with attributes"
                        )
                        return value, attributes
                    else:
                        logger.debug(
                            f"Successfully retrieved key: {key} from bin_id: {bin_id}"
                        )
                        return value
                else:
                    logger.debug(f"Key '{key}' not found in bin_id: {bin_id}")
                    raise KeyError(f"Key '{key}' not found in cache.")

    def __setitem__(self, key, value):
        """Set a value in the cache, ensuring only one process writes at a time."""
        bin_id = self._get_bin_id(key)
        bin_path = self._get_bin_path(bin_id)
        lock = self.locks[bin_id]

        if isinstance(value, tuple):
            value, attributes = value
            logger.debug("Attempting to save value together with attributes")
        else:
            attributes = {}

        logger.debug(f"Attempting to set key: {key} in bin_id: {bin_id}")
        with lock:  # Acquire lock before writing
            with h5py.File(bin_path, "a") as f:
                if key in f:
                    del f[key]  # Delete existing dataset if it exists
                    logger.debug(f"Deleted existing key: {key} in bin_id: {bin_id}")
                ds = f.create_dataset(key, data=np.array(value))
                ds.attrs.update(attributes)
                logger.debug(f"Successfully set key: {key} in bin_id: {bin_id}")

    def __delitem__(self, key):
        """Delete a value from the cache."""
        bin_id = self._get_bin_id(key)
        bin_path = self._get_bin_path(bin_id)
        lock = self.locks[bin_id]

        logger.debug(f"Attempting to delete key: {key} from bin_id: {bin_id}")
        with lock:  # Acquire lock before deleting
            with h5py.File(bin_path, "a") as f:
                if key in f:
                    del f[key]
                    logger.debug(
                        f"Successfully deleted key: {key} from bin_id: {bin_id}"
                    )
                else:
                    logger.debug(f"Key '{key}' not found in bin_id: {bin_id}")
                    raise KeyError(f"Key '{key}' not found in cache.")

    def __contains__(self, key):
        """Check if a key exists in the cache."""
        bin_id = self._get_bin_id(key)
        bin_path = self._get_bin_path(bin_id)
        lock = self.locks[bin_id]

        logger.debug(f"Checking if key: {key} exists in bin_id: {bin_id}")
        with lock:  # Acquire lock before reading
            if not bin_path.exists():
                logger.debug(
                    f"Bin file for bin_id: {bin_id} does not exist, so key does not exist."
                )
                return False
            with h5py.File(bin_path, "r") as f:
                exists = key in f
                logger.debug(f"Key: {key} exists: {exists} in bin_id: {bin_id}")
                return exists

    def get(self, key, default=None):
        """Retrieve a value from the cache, returning a default value if not found."""
        try:
            value = self[key]
            logger.debug(f"Retrieved key: {key} from cache")
            return value
        except KeyError:
            logger.debug(f"Key: {key} not found, returning default value: {default}")
            return default

    def set(self, key, value):
        """Set a value in the cache."""
        self[key] = value

    def delete(self, key):
        """Delete a value from the cache."""
        del self[key]

    def clear(self):
        """Clear the entire cache."""
        logger.debug("Clearing the entire cache")
        for bin_id in range(self.num_bins):
            lock = self.locks[bin_id]
            bin_path = self._get_bin_path(bin_id)

            with lock:  # Acquire lock before clearing
                if bin_path.exists():
                    bin_path.unlink()
                    logger.debug(
                        f"Cleared bin_id: {bin_id} by deleting bin_path: {bin_path}"
                    )

    def __len__(self):
        """Return the total number of items in the cache."""
        total_length = 0
        logger.debug("Calculating the total number of items in the cache")
        for bin_id in range(self.num_bins):
            bin_path = self._get_bin_path(bin_id)
            lock = self.locks[bin_id]

            with lock:  # Acquire lock before reading
                if not bin_path.exists():
                    logger.debug(f"Bin file for bin_id: {bin_id} does not exist.")
                    continue
                with h5py.File(bin_path, "r") as f:
                    total_length += len(f.keys())
                    logger.debug(f"Counted {len(f.keys())} items in bin_id: {bin_id}")

        logger.debug(f"Total number of items in cache: {total_length}")
        return total_length

    def __iter__(self):
        """Iterate over all keys in the cache."""
        logger.debug("Iterating over all keys in the cache")
        for bin_id in range(self.num_bins):
            bin_path = self._get_bin_path(bin_id)
            lock = self.locks[bin_id]

            with lock:  # Acquire lock before reading
                if not bin_path.exists():
                    logger.debug(f"Bin file for bin_id: {bin_id} does not exist.")
                    continue
                with h5py.File(bin_path, "r") as f:
                    for key in f.keys():
                        logger.debug(f"Yielding key: {key} from bin_id: {bin_id}")
                        yield key
