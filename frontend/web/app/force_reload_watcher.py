import os
import time
import random

from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

APP_PATH = "app.py"
WATCH_DIRS = ["pages", "components"]
VALID_EXT = [".py"]


class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and any(
            event.src_path.endswith(ext) for ext in VALID_EXT
        ):
            print(f"[✔] Cambio detectado en: {event.src_path}")
            append_random_hashes(APP_PATH)


def append_random_hashes(filepath):
    num_hashes = random.randint(1, 100)
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n{'#' * num_hashes}\n")
        print(f"[📎] Modificado {filepath} con {'#' * num_hashes}")
    except Exception as e:
        print(f"[❌] Error modificando {filepath}: {e}")


if __name__ == "__main__":
    observer = Observer(timeout=1)  # más sensible, menos latencia
    for directory in WATCH_DIRS:
        abs_path = os.path.abspath(directory)
        if os.path.isdir(abs_path):
            observer.schedule(ChangeHandler(), path=abs_path, recursive=True)
            print(f"[🕵️‍♂️] Observando: {abs_path}")
        else:
            print(f"[⚠] Carpeta no encontrada: {abs_path}")

    observer.start()
    print("[🔁] Watcher activo. Ctrl+C para detener.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[⏹] Watcher detenido.")

    observer.join()
