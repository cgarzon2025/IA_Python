import onnx
import numpy as np
import onnxruntime as ort

def verify_model():
    try:
        # Cargar el modelo
        print("Cargando modelo ONNX...")
        model = onnx.load("carpeta_salida/model.onnx")
        
        # Verificar el modelo
        print("Verificando modelo...")
        onnx.checker.check_model(model)
        print("Modelo verificado correctamente")
        
        # Mostrar información del modelo
        print("\nInformación del modelo:")
        print(f"Versión de IR: {model.ir_version}")
        print(f"Productor: {model.producer_name}")
        print(f"Versión del productor: {model.producer_version}")
        
        # Mostrar información de las entradas
        print("\nEntradas del modelo:")
        for input in model.graph.input:
            print(f"Nombre: {input.name}")
            print(f"Tipo: {input.type.tensor_type.elem_type}")
            print(f"Forma: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        
        # Mostrar información de las salidas
        print("\nSalidas del modelo:")
        for output in model.graph.output:
            print(f"Nombre: {output.name}")
            print(f"Tipo: {output.type.tensor_type.elem_type}")
            print(f"Forma: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
        
        # Probar el modelo con un valor de ejemplo
        print("\nProbando el modelo con un valor de ejemplo...")
        session = ort.InferenceSession("carpeta_salida/model.onnx")
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Crear un tensor de entrada
        input_data = np.array([[20.0]], dtype=np.float32)
        print(f"Entrada: {input_data}")
        
        # Ejecutar el modelo
        outputs = session.run([output_name], {input_name: input_data})
        print(f"Salida: {outputs[0]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    verify_model() 