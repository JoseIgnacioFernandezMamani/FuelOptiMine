import streamlit as st
import numpy as np
import pandas as pd
from streamlit_elements import elements, dashboard, mui


@st.cache_data()
def get_data():
    df = pd.DataFrame(
        np.random.randint(0, 100, 50).reshape(-1, 5),
        columns=list("abcde"),
    )
    return df


def supply_eda_page_with_datagrid():
    st.title("Dashboard con DataGrid usando streamlit-elements y MUI")

    data = get_data()

    layout = [
        dashboard.Item("datagrid", 0, 0, 12, 6),
    ]

    with elements("dashboard_datagrid"):
        with dashboard.Grid(layout, draggableHandle=".draggable"):
            with mui.Card(
                key="datagrid", sx={"display": "flex", "flexDirection": "column"}
            ):
                mui.CardHeader(title="DataGrid Editable", className="draggable")
                with mui.CardContent(sx={"flex": 1, "minHeight": 0}):
                    rows = data.to_dict(orient="records")
                    columns = [
                        {
                            "field": col,
                            "headerName": col.upper(),
                            "width": 120,
                            "editable": True,
                        }
                        for col in data.columns
                    ]

                    mui.DataGrid(
                        rows=rows,
                        columns=columns,
                        pageSize=10,
                        rowsPerPageOptions=[5, 10, 20],
                        checkboxSelection=True,
                        disableSelectionOnClick=True,
                        autoHeight=True,
                        experimentalFeatures={"newEditingApi": True},
                    )

    st.info(
        """
        - Puedes arrastrar y redimensionar la tarjeta del DataGrid.
        - Las celdas son editables directamente en la tabla.
        - La selección múltiple está habilitada.
        """
    )


# Ejecutar la función directamente al importar el módulo para que st.Page la muestre
supply_eda_page_with_datagrid()
