# diagrams

## system context diagram

Definir el alcance global del sistema, identificando los limites, actores externos y entradas/salidas relevantes para el consumo de diesel en operaciones de extraccion minera.

```mermaid
graph TD
    %% === Entidades Externas (Interactúan directamente) ===
    A[Despachadores /<br>Supervisores] -->|Asignación de Rutas<br>Priorización de Tareas| B((Operaciones de Extracción Minera))
    C[Operadores] -->|Ejecución de Actividades<br>Reporte de Incidencias| B
    D[Mantenimiento] -->|Disponibilidad de Equipos<br>Estado Técnico| B
    E[Rutas y Condiciones] -->|Buenas condiciones de <br>accesibilidad de Vías| B
    F[Clima] -->|Lluvia, Niebla, Temperatura| B
    
    %% === Sistema Central ===
    B -->|Material Extraído| G[Procesamiento de Planta y Botaderos]
    B -->|Demandas Operativas| A
    B -->|Solicitudes de Reparación| D
    B -->|Requerimiento de Ajustes de Ruta| E
```
