# Laboratorio 3 - Preprocesamiento de Colección CACM

Este laboratorio implementa el parseo y preprocesamiento de la colección CACM (Communications of the ACM) para tareas de Recuperación de Información.

## Estructura del Proyecto

```
Lab3/
├── main.py                 # Script principal
├── README.md               # Este archivo
├── resources/
│   └── CACM/
│       ├── cacm.all        # Documentos de la colección
│       ├── query.text      # Consultas de prueba
│       ├── qrels.text      # Relevancia de documentos por consulta
│       ├── common_words    # Stopwords
│       ├── cite.info       # Información de citaciones
│       └── README          # Documentación original
└── output/
    ├── docs/               # Documentos preprocesados
    │   ├── documentos_parseados.txt
    │   ├── documentos_tokens.txt
    │   ├── documentos_stripped.txt
    │   ├── documentos_lowercase.txt
    │   ├── documentos_no_stopwords.txt
    │   ├── documentos_porter.txt
    │   ├── documentos_snowball.txt
    │   └── documentos_lemmatized.txt
    └── queries/            # Consultas preprocesadas
        ├── consultas_parseadas.txt
        ├── consultas_tokens.txt
        ├── consultas_stripped.txt
        ├── consultas_lowercase.txt
        ├── consultas_no_stopwords.txt
        ├── consultas_porter.txt
        ├── consultas_snowball.txt
        └── consultas_lemmatized.txt
```

## Formato de Archivos CACM

### Etiquetas en documentos (`cacm.all`)

| Etiqueta | Significado | Descripción |
|----------|-------------|-------------|
| `.I` | ID | Número único del documento |
| `.T` | Title | Título del artículo |
| `.W` | Words/Abstract | Resumen del documento (opcional) |
| `.B` | Bibliographic | Información bibliográfica (revista y fecha) |
| `.A` | Authors | Lista de autores |
| `.N` | Notes | Notas internas del sistema |
| `.X` | Cross-references | Información de citaciones |
| `.K` | Keywords | Palabras clave (si las hay) |

### Etiquetas en consultas (`query.text`)

| Etiqueta | Significado | Descripción |
|----------|-------------|-------------|
| `.I` | ID | Número único de la consulta |
| `.W` | Words | Texto de la consulta |
| `.A` | Authors | Autores buscados (opcional) |
| `.N` | Notes | Quién realizó la consulta |

## Estadísticas de la Colección

| Elemento | Cantidad |
|----------|----------|
| Documentos | 3,204 |
| Consultas | 64 |


## Dependencias

- **NLTK** - Natural Language Toolkit
  - `punkt_tab` - Tokenización
  - `wordnet` - Lematización
  - `stopwords` - Palabras vacías

