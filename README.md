# Lab 15: Base de Datos Vectorial - Tecsup

Este proyecto implementa una base de datos vectorial con información personal de estudiantes utilizando `chromadb`. Se compara el rendimiento de dos motores de embeddings: el predeterminado de Chroma y `sentence-transformers`.

---

## 🧠 Objetivo

Desarrollar una base de datos vectorial llamada **"Tecsup"** con datos personales y académicos. El sistema debe:

* Permitir consultas semánticas.
* Comparar resultados entre embeddings por defecto y embeddings personalizados.

---

## 🔧 Requisitos

Instala las dependencias con:

```bash
pip install chromadb sentence-transformers pytest
```

---

## 📁 Estructura del Proyecto

```
.
├── utils.py               # Funciones reutilizables
├── test_utils.py          # Tests con pytest
├── main.py (opcional)     # Lógica principal del programa
├── README.md              # Este archivo
```

---

## 🚀 Cómo Ejecutar

### 1. Ejecutar el código principal

Puedes ejecutar todo en un entorno como Jupyter Notebook o crear un `main.py` que llame a las funciones desde `utils.py`.

### 2. Correr los tests

```bash
pytest test_utils.py -v
```

---

## 🔍 Funcionalidades

* Crear base de datos vectorial con `chromadb`
* Agregar documentos personales
* Consultar datos con preguntas naturales (ej: "¿Quién es beneficiario de BECA 18?")
* Utilizar motores de embeddings:

  * `chromadb` default
  * `sentence-transformers` (`all-MiniLM-L6-v2`)
* Comparar respuestas de ambos métodos

---

## 📊 Ejemplo de Consulta

```python
¿Quién es beneficiario de BECA 18?
```

**Respuesta con chromadb default**:

> Kevin Estiben Olortegui Perez es un beneficiario de BECA 18

**Respuesta con embeddings personalizados**:

> Kevin Estiben Olortegui Perez es un beneficiario de BECA 18

---

## 🧪 Testing

Incluye pruebas unitarias para:

* Creación de colección
* Embedding manual de documentos
* Consulta semántica precisa

---

## 🧑‍💻 Autor

* Kevin Estiben Olortegui Perez

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. ¡Libre de usar y modificar!
