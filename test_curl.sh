#!/bin/bash

# Script para probar la API con curl
# Asegúrate de que el servidor esté ejecutándose: python run_server.py

API_KEY="dev-api-key-12345"
BASE_URL="http://localhost:8000"

echo "🧪 Probando ML API Platform con curl"
echo "===================================="

# Test health endpoints
echo "🏥 Health checks:"
echo "GET /health/ready"
curl -s "$BASE_URL/health/ready" | jq '.'
echo

echo "GET /health/live"
curl -s "$BASE_URL/health/live" | jq '.'
echo

# Test prediction without auth (should fail)
echo "🔒 Predicción sin autenticación (debería fallar):"
echo "POST /predict (sin auth)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"feature1": 1.5, "feature2": "test", "feature3": 42}
    ]
  }' | jq '.'
echo

# Test prediction with auth
echo "✅ Predicción con autenticación:"
echo "POST /predict (con auth)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "data": [
      {
        "feature1": 1.5,
        "feature2": "category_a", 
        "feature3": 42,
        "feature4": 0.8
      },
      {
        "feature1": 2.3,
        "feature2": "category_b",
        "feature3": 37, 
        "feature4": 1.2
      }
    ]
  }' | jq '.'
echo

# Test validation error
echo "⚠️  Error de validación (datos vacíos):"
echo "POST /predict (datos vacíos)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"data": []}' | jq '.'
echo

# Test metrics
echo "📊 Métricas:"
echo "GET /metrics"
curl -s "$BASE_URL/metrics" \
  -H "Authorization: Bearer $API_KEY" | jq '.'
echo

echo "🎉 ¡Pruebas completadas!"