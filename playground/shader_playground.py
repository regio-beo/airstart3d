from vpython import *

scene.title = "Custom WebGL Shader in VPython"
scene.background = color.black

# Create a sphere
glow_sphere = sphere(pos=vector(0,0,0), radius=1, color=color.white)

# Inject WebGL Shader via JavaScript
js_code = """

let gl = document.getElementsByTagName("canvas")[0].getContext("webgl2");  // Access VPython’s WebGL context
console.log('gl', gl);
let program = gl.createProgram();

// Vertex Shader (Passes Position & Color)
let vertexShaderSource = `
    attribute vec4 a_position;
    void main() {
        gl_Position = a_position;
    }
`;
let vertexShader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vertexShader, vertexShaderSource);
gl.compileShader(vertexShader);
gl.attachShader(program, vertexShader);

// Fragment Shader (Glow Effect)
let fragmentShaderSource = `
    precision mediump float;
    uniform vec2 resolution;
    void main() {
        vec2 st = gl_FragCoord.xy / resolution;
        gl_FragColor = vec4(st.x, 0.0, st.y, 1.0); // Gradient Shader
    }
`;
let fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fragmentShader, fragmentShaderSource);
gl.compileShader(fragmentShader);
gl.attachShader(program, fragmentShader);

// Link and Use Shader Program
gl.linkProgram(program);
gl.useProgram(program);
"""


scene.append_to_caption("Run this: <script>" + js_code + "</script>")  # Run JavaScript in VPython
