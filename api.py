from flask import Flask, jsonify, request
import replicate

app = Flask(__name__)

# Config Replicate: https://replicate.com/
# export REPLICATE_API_TOKEN=xxxxxxxxxxxxxx

@app.route('/generate', methods=['POST'])
def generate_images():
    model = replicate.models.get("borisdayma/dalle-mini")
    version = model.versions.get("2e3975b1692cd6aecac28616dba364cc9f1e30c610c6efd62dbe9b9c7d1d03ea")
    prompt = request.json.get('prompt', '')
    n_predictions = int(request.json.get('n_predictions', 1))
    show_clip_score = bool(request.json.get('show_clip_score', False))

    inputs = {
        'prompt': prompt,
        'n_predictions': n_predictions,
        'show_clip_score': show_clip_score,
    }

    output = version.predict(**inputs)
    return jsonify(output)

@app.route('/generate-anime', methods=['POST'])
def generate_images_anime():
    model = replicate.models.get("cjwbw/pastel-mix")
    version = model.versions.get("0c9ff376fe89e11daecf5a3781d782acc69415b2f1fa910460c59e5325ed86f7")
    prompt = request.json.get('prompt', '')
    negative_prompt = request.json.get('negative_prompt', None)
    width = request.json.get('width', 512)
    height = request.json.get('height', 512)
    num_outputs = request.json.get('num_outputs', 1)
    num_inference_steps = request.json.get('num_inference_steps', 50)
    guidance_scale = request.json.get('guidance_scale', 12)
    scheduler = request.json.get('scheduler', 'DPMSolverMultistep')
    seed = request.json.get('seed', None)

    inputs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'num_outputs': num_outputs,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'scheduler': scheduler,
        'seed': seed,
    }

    output = version.predict(**inputs)

    return jsonify(output)

@app.route('/generate-vip', methods=['POST'])
def generate_images_vip():
    model = replicate.models.get("stability-ai/stable-diffusion")
    version = model.versions.get("db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")
    prompt = request.json.get('prompt', '')
    image_dimensions = request.json.get('image_dimensions', '768x768')
    negative_prompt = request.json.get('negative_prompt', None)
    num_outputs = request.json.get('num_outputs', 1)
    num_inference_steps = request.json.get('num_inference_steps', 50)
    guidance_scale = request.json.get('guidance_scale', 7.5)
    scheduler = request.json.get('scheduler', 'DPMSolverMultistep')
    seed = request.json.get('seed', None)

    width, height = map(int, image_dimensions.split('x'))

    inputs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'num_outputs': num_outputs,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'scheduler': scheduler,
        'seed': seed,
    }

    output = version.predict(**inputs)

    return jsonify(output)

@app.route('/generate-midjourney', methods=['POST'])
def generate_images_midjourney():
    model = replicate.models.get("tstramer/midjourney-diffusion")
    version = model.versions.get("436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b")
    prompt = request.json.get('prompt', '')
    negative_prompt = request.json.get('negative_prompt', None)
    width = request.json.get('width', 768)
    height = request.json.get('height', 768)
    prompt_strength = request.json.get('prompt_strength', 0.8)
    num_outputs = request.json.get('num_outputs', 1)
    num_inference_steps = request.json.get('num_inference_steps', 50)
    guidance_scale = request.json.get('guidance_scale', 7.5)
    scheduler = request.json.get('scheduler', 'DPMSolverMultistep')
    seed = request.json.get('seed', None)

    # Tạo các tham số đầu vào cho hàm predict
    inputs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'prompt_strength': prompt_strength,
        'num_outputs': num_outputs,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'scheduler': scheduler,
        'seed': seed,
    }

    output = version.predict(**inputs)

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
