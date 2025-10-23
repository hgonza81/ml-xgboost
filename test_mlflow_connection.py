#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión con MLflow.
"""

import sys
import os

# Agregar el directorio src al path
sys.path.append('src')

# Configurar la URI de MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'

try:
    from mlflow_components import (
        get_tracking_server, 
        get_model_registry, 
        get_experiment_manager,
        get_mlflow_health_check
    )
    
    print("✅ Importación de componentes MLflow exitosa")
    
    # Probar health check
    health_check = get_mlflow_health_check()
    health_result = health_check.comprehensive_health_check()
    
    print(f"✅ Health check completado:")
    print(f"   Estado general: {health_result['overall_status']}")
    
    for component, status in health_result['components'].items():
        print(f"   {component}: {status['status']}")
    
    # Probar creación de experimento
    experiment_manager = get_experiment_manager()
    experiment_id = experiment_manager.create_experiment(
        name="test_connection",
        tags={"purpose": "connection_test"}
    )
    
    print(f"✅ Experimento de prueba creado: {experiment_id}")
    
    # Probar registro de modelo (simulado)
    model_registry = get_model_registry()
    registered_models = model_registry.list_registered_models()
    
    print(f"✅ Conexión al registro de modelos exitosa")
    print(f"   Modelos registrados: {len(registered_models)}")
    
    print("\n🎉 ¡Todas las pruebas de conexión MLflow exitosas!")
    
except Exception as e:
    print(f"❌ Error en la prueba de conexión: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)