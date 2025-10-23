#!/usr/bin/env python3
"""
Script para probar los endpoints de la API en local.
"""

import requests
import json
import time

# Configuración
BASE_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_health_endpoints():
    """Probar endpoints de salud."""
    print("🏥 Probando endpoints de salud...")
    
    # Test readiness
    response = requests.get(f"{BASE_URL}/health/ready")
    print(f"   GET /health/ready: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Ready: {response.json()['status']}")
    
    # Test liveness
    response = requests.get(f"{BASE_URL}/health/live")
    print(f"   GET /health/live: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Alive: {response.json()['status']}")
    print()

def test_prediction_endpoint():
    """Probar endpoint de predicción."""
    print("🔮 Probando endpoint de predicción...")
    
    # Datos de prueba
    test_data = {
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
            },
            {
                "feature1": 0.9,
                "feature2": "category_c",
                "feature3": 55,
                "feature4": 0.6
            }
        ]
    }
    
    # Test sin autenticación
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"   POST /predict (sin auth): {response.status_code}")
    if response.status_code in [401, 403]:
        print("   ✅ Autenticación requerida correctamente")
    
    # Test con autenticación
    response = requests.post(f"{BASE_URL}/predict", json=test_data, headers=HEADERS)
    print(f"   POST /predict (con auth): {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Predicciones: {result['predictions']}")
        print(f"   📊 Muestras procesadas: {result['samples_processed']}")
        print(f"   🏷️  Versión del modelo: {result['model_version']}")
        print(f"   ⏰ Timestamp: {result['timestamp']}")
        if 'model_metadata' in result and result['model_metadata']:
            metadata = result['model_metadata']
            print(f"   🤖 Metadatos del modelo:")
            print(f"      - Nombre: {metadata.get('name', 'N/A')}")
            print(f"      - Versión: {metadata.get('version', 'N/A')}")
            print(f"      - Etapa: {metadata.get('stage', 'N/A')}")
            print(f"      - Run ID: {metadata.get('run_id', 'N/A')}")
    elif response.status_code == 503:
        print("   ⚠️  Servicio de modelo no disponible (esperado si MLflow no está configurado)")
        error = response.json()
        print(f"   📝 Detalle: {error.get('detail', 'N/A')}")
    else:
        print(f"   ❌ Error: {response.text}")
    print()

def test_validation_errors():
    """Probar validación de errores."""
    print("⚠️  Probando validación de errores...")
    
    # Test con datos vacíos
    empty_data = {"data": []}
    response = requests.post(f"{BASE_URL}/predict", json=empty_data, headers=HEADERS)
    print(f"   POST /predict (datos vacíos): {response.status_code}")
    
    if response.status_code == 400:
        error = response.json()
        print(f"   ✅ Error de validación: {error['error']}")
        print(f"   📝 Mensaje: {error['message']}")
    
    # Test con tipos incorrectos
    invalid_data = {
        "data": [
            {"feature1": [1, 2, 3]}  # Lista no permitida
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data, headers=HEADERS)
    print(f"   POST /predict (tipos inválidos): {response.status_code}")
    
    if response.status_code == 400:
        print("   ✅ Tipos inválidos rechazados correctamente")
    print()

def test_model_info_endpoint():
    """Probar endpoint de información del modelo."""
    print("🤖 Probando endpoint de información del modelo...")
    
    response = requests.get(f"{BASE_URL}/model/info", headers=HEADERS)
    print(f"   GET /model/info: {response.status_code}")
    
    if response.status_code == 200:
        model_info = response.json()
        print(f"   ✅ Servicio: {model_info['service']}")
        print(f"   🏷️  Modelo configurado: {model_info['configured_model']['name']}")
        print(f"   🎯 Etapa: {model_info['configured_model']['stage']}")
        if 'model_info' in model_info:
            info = model_info['model_info']
            print(f"   📊 Información del modelo:")
            print(f"      - Nombre: {info.get('name', 'N/A')}")
            print(f"      - Versión: {info.get('version', 'N/A')}")
            print(f"      - Etapa: {info.get('stage', 'N/A')}")
    elif response.status_code == 503:
        print("   ⚠️  Modelo no disponible (esperado si MLflow no está configurado)")
        error = response.json()
        print(f"   📝 Detalle: {error.get('detail', 'N/A')}")
    else:
        print(f"   ❌ Error: {response.text}")
    print()


def test_metrics_endpoint():
    """Probar endpoint de métricas."""
    print("📊 Probando endpoint de métricas...")
    
    response = requests.get(f"{BASE_URL}/metrics", headers=HEADERS)
    print(f"   GET /metrics: {response.status_code}")
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"   ✅ Servicio: {metrics['service']}")
        print(f"   📈 Métricas:")
        for key, value in metrics['metrics'].items():
            print(f"      - {key}: {value}")
    else:
        print(f"   ❌ Error: {response.text}")
    print()

def main():
    """Función principal para ejecutar todas las pruebas."""
    print("🧪 Probando ML API Platform")
    print("=" * 50)
    
    try:
        # Verificar que el servidor esté ejecutándose
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ El servidor no está respondiendo correctamente")
            return
        
        print("✅ Servidor detectado y funcionando")
        print()
        
        # Ejecutar pruebas
        test_health_endpoints()
        test_prediction_endpoint()
        test_validation_errors()
        test_model_info_endpoint()
        test_metrics_endpoint()
        
        print("🎉 ¡Todas las pruebas completadas!")
        
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor.")
        print("   Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
        print("   Ejecuta: python run_server.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()