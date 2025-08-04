from openpose_gpu.op_python_wrapper import run_openpose
from segformer_model import SegformerModel
seg_model = SegformerModel()


person_path = "main_inputs/user_inputs/person"
output_path = "main_inputs/generated_inputs"

run_openpose(
    person_dir=person_path, 
    output_dir=output_path
)