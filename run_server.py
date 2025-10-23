#!/usr/bin/env python3
"""
Script para ejecutar el servidor FastAPI en desarrollo.
"""

import uvicorn
import os

if __name__ == "__main__":
    # Configurar variables de entorno para desarrollo
    os.environ.setdefault("API_KEY", "dev-api-key-12345")
    os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    print("🚀 Iniciando ML API Platform...")
    print("📋 Configuración:")
    print(f"   - API Key: {os.environ['API_KEY']}")
    print(f"   - CORS Origins: {os.environ['CORS_ORIGINS']}")
    print(f"   - Log Level: {os.environ['LOG_LEVEL']}")
    print()
    print("📖 Documentación disponible en:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print()
    print("🔍 Endpoints disponibles:")
    print("   - GET  /health/ready - Health check")
    print("   - GET  /health/live  - Liveness check")
    print("   - POST /predict      - Predicciones (requiere auth)")
    print("   - GET  /metrics      - Métricas (requiere auth)")
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recarga automática en desarrollo
        log_level="info"
    )