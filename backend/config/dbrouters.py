from clickhouse_backend.models import ClickhouseModel


def get_subclasses(class_):
    classes = class_.__subclasses__()
    index = 0
    while index < len(classes):
        classes.extend(classes[index].__subclasses__())
        index += 1
    return list(set(classes))


class ClickHouseRouter:
    def __init__(self):
        self.route_model_names = set()
        for model in get_subclasses(ClickhouseModel):
            if model._meta.abstract:
                continue
            self.route_model_names.add(model._meta.label_lower)

    def db_for_read(self, model, **hints):
        if model._meta.label_lower in self.route_model_names or hints.get("clickhouse"):
            return "clickhouse"
        return None

    def db_for_write(self, model, **hints):
        if model._meta.label_lower in self.route_model_names or hints.get("clickhouse"):
            return "clickhouse"
        return None

    def allow_migrate(self, db, app_label, **hints):
        if app_label == "analytics":
            return db == "clickhouse"
        # Todas las demás apps usan la BD default (SQLite)
        return db == "default"
