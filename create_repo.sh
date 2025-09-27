#!/bin/bash
# Este script prepara un repositorio limpio para el backend de informes.

# Detiene el script si cualquier comando falla
set -e

# --- CONFIGURACIÓN ---
# Ruta a tu repositorio geocore original (ajusta si no está en tu carpeta de usuario)
ORIGINAL_REPO="$HOME/geocore" 

# Ruta y nombre para tu nuevo repositorio limpio
NEW_REPO="$HOME/twolapsreport"
# --- FIN DE LA CONFIGURACIÓN ---

echo ">> Creando el nuevo repositorio en: $NEW_REPO"

# 1) Crea la estructura de carpetas
echo "✅ 1/8: Creando estructura de directorios..."
mkdir -p "$NEW_REPO"/backend/src/reports
mkdir -p "$NEW_REPO"/backend/src/scheduler
mkdir -p "$NEW_REPO"/backend/src/engines
mkdir -p "$NEW_REPO"/backend/src/insight_analysis
mkdir -p "$NEW_REPO"/backend/src/utils
mkdir -p "$NEW_REPO"/backend/src/db
mkdir -p "$NEW_REPO"/backend/scripts
mkdir -p "$NEW_REPO"/backend/files

# 2) Copia los módulos necesarios del backend
echo "✅ 2/8: Copiando módulos del backend..."
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/reports/ "$NEW_REPO"/backend/src/reports/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/scheduler/ "$NEW_REPO"/backend/src/scheduler/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/engines/ "$NEW_REPO"/backend/src/engines/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/insight_analysis/ "$NEW_REPO"/backend/src/insight_analysis/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/utils/ "$NEW_REPO"/backend/src/utils/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/src/db/ "$NEW_REPO"/backend/src/db/
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORIGINAL_REPO"/backend/scripts/ "$NEW_REPO"/backend/scripts/

# 3) Copia los archivos raíz importantes
echo "✅ 3/8: Copiando archivos de configuración..."
cp "$ORIGINAL_REPO"/backend/requirements.txt "$NEW_REPO"/backend/
cp "$ORIGINAL_REPO"/docker-compose.yml "$NEW_REPO"/backend/
cp "$ORIGINAL_REPO"/backend/app.py "$NEW_REPO"/backend/

# 4) Copia backups de la base de datos (si existen)
echo "✅ 4/8: Copiando backups de BD (si existen)..."
cp "$ORIGINAL_REPO"/backend/geocore_backup.sql "$NEW_REPO"/backend/ 2>/dev/null || true

# 5) Crea un .env.example limpio
echo "✅ 5/8: Creando .env.example..."
cat > "$NEW_REPO"/backend/.env.example << 'EOF'
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=ai_visibility
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

OPENAI_API_KEY=your_openai_key
SERPAPI_KEY=your_serpapi_key
PERPLEXITY_API_KEY=your_perplexity_key
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TIMEOUT=25

SLACK_WEBHOOK_URL=
DEFAULT_BRAND=Twolaps
EOF

# 6) Crea un .gitignore mínimo
echo "✅ 6/8: Creando .gitignore..."
cat > "$NEW_REPO"/.gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
*.log
.DS_Store
EOF

# 7) Prepara la carpeta 'files'
echo "✅ 7/8: Preparando la carpeta 'files'..."
touch "$NEW_REPO"/backend/files/.gitkeep

# 8) Inicializa el nuevo repositorio Git
echo "✅ 8/8: Inicializando el nuevo repositorio Git..."
cd "$NEW_REPO"
git init
git add .
git commit -m "Extract: backend minimal para informes y polling"

echo ""
echo "🎉 ¡Listo! El nuevo repositorio 'twolapsreport' se ha creado en '$NEW_REPO'."