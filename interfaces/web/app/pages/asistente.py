import streamlit as st

st.title("🤖 Asistente de IA")
st.markdown("Consulta información usando lenguaje natural")

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("¿Cómo puedo ayudarte hoy?"):
    # Simular respuesta de IA
    respuesta = f"**Respuesta:** Para '{prompt}', recomiendo revisar la ruta Norte con 3 camiones."
    
    # Actualizar historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    
    # Mostrar nueva interacción
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(respuesta)