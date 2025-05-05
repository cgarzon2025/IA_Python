import tensorflow as tf
import tf2onnx
import onnx
import os

try:
    # Cargar el modelo HDF5 con compile=False y custom_objects
    print("Cargando modelo HDF5...")
    model = tf.keras.models.load_model('celsius_to_fahrenheit.h5', 
                                     compile=False,
                                     custom_objects={'optimizer': tf.keras.optimizers.legacy.Adam})
    print("Modelo cargado exitosamente")
    
    # Recompilar el modelo con un optimizador legacy
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
    print("Modelo recompilado")
    
    print(f"Arquitectura del modelo: {model.summary()}")
    
    # Definir la especificación de entrada
    print("Definiendo especificación de entrada...")
    input_signature = [tf.TensorSpec([1, 1], tf.float32, name="input")]
    
    # Convertir a ONNX
    print("Convirtiendo a ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=input_signature,
        opset=13
    )
    print("Conversión a ONNX completada")
    
    # Asegurarse de que el directorio existe
    os.makedirs("carpeta_salida", exist_ok=True)
    
    # Guardar el modelo ONNX
    print("Guardando modelo ONNX...")
    output_path = os.path.join("carpeta_salida", "model.onnx")
    onnx.save(model_proto, output_path)
    print(f"Modelo guardado en: {output_path}")
    
    print("¡Conversión completada con éxito!")
    
except Exception as e:
    print(f"Error durante la conversión: {str(e)}")
    print(f"Tipo de error: {type(e)}")
    import traceback
    print("Traceback completo:")
    print(traceback.format_exc()) 