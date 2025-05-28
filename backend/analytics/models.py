from clickhouse_backend import models as ch_models


class PruebaConexion(ch_models.ClickhouseModel):
    mensaje = ch_models.StringField()
    fecha = ch_models.DateTime64Field(auto_now_add=True)

    class Meta:
        engine = ch_models.engines.MergeTree(order_by=("fecha",))
        db_table = "prueba_conexion"
