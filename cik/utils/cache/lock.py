import time
import logging
from pathlib import Path

logger = logging.getLogger("Directory Lock")


class DirLock:
    """
    A directory-based locking mechanism to ensure only one process can access a resource at a time.
    The lock is acquired by creating a directory.

    Note: We rely on the atomicity of directory creation to make a locking mechanism that is compatible
          with NFS since standard locking mechanisms like fcntl and flock are not supported on NFS.

    Parameters:
    -----------
    lock_dir : str
        The directory path to use for the lock.
    retry_interval : float, default 0.1
        The time in seconds to wait before retrying to acquire the lock.
    timeout : float, default None
        The maximum time in seconds to wait for acquiring the lock. If None, it will wait indefinitely.

    """

    def __init__(self, lock_dir, retry_interval=0.1, timeout=None):
        self.lock_dir = Path(lock_dir)
        self.retry_interval = retry_interval
        self.timeout = timeout

    def acquire(self):
        """Acquire the directory-based lock."""
        logging.debug(f"Attempting to acquire lock at {self.lock_dir}")
        start_time = time.time()

        while True:
            try:
                # Attempt to create the directory to acquire the lock
                self.lock_dir.mkdir(parents=True, exist_ok=False)
                logging.debug(f"Lock acquired at {self.lock_dir}")
                return
            except FileExistsError:
                elapsed_time = time.time() - start_time
                if self.timeout is not None and elapsed_time >= self.timeout:
                    raise TimeoutError(
                        f"Failed to acquire lock within {self.timeout} seconds."
                    )

                # XXX: Caveat: Due to race conditions, the lock age can sometimes be off.
                logging.debug(
                    f"Lock is not available at {self.lock_dir} (age: {self.age()}), waiting {self.retry_interval} seconds."
                )
                time.sleep(self.retry_interval)

    def release(self):
        """Release the directory-based lock."""
        logging.debug(f"Attempting lock release at {self.lock_dir}")
        try:
            self.lock_dir.rmdir()  # Remove the lock directory
            logging.debug(f"Lock released at {self.lock_dir}")
        except OSError as e:
            raise RuntimeError(f"Failed to release lock: {e}")

    def age(self):
        """
        Return the age of the lock in seconds since its creation.

        Returns:
        --------
        age_seconds : float
            The age of the lock in seconds and None if the lock directory does not exist.

        """
        try:
            # Get the creation time of the lock directory
            creation_time = self.lock_dir.stat().st_mtime
            current_time = time.time()
            age_seconds = current_time - creation_time
            return age_seconds
        except FileNotFoundError:
            logging.debug(f"Lock directory {self.lock_dir} does not exist.")
            return None

    def __enter__(self):
        """Acquire the lock when entering the context."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the lock when exiting the context."""
        self.release()
