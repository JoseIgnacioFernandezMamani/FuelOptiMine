import os
import sys
from django.core.management import execute_from_command_line


def setup():
    # Aplicar migraciones de Django (auth, admin, etc.)
    execute_from_command_line([sys.argv[0], "migrate"])

    # Crear superusuario inicial (si no existe)
    from django.contrib.auth import get_user_model

    User = get_user_model()
    if not User.objects.filter(username="admin").exists():
        User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="admin",  # Contraseña temporal (cambiar en producción)
        )
        print("✅ Superusuario creado: admin / admin")


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    setup()
