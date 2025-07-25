#version 450

layout(location = 0) in vec2 fragTexCoord;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

vec3 colormap(float value) {
    // Create a colorful visualization
    vec3 c1 = vec3(0.0, 0.0, 0.5);
    vec3 c2 = vec3(0.0, 0.5, 1.0);
    vec3 c3 = vec3(0.5, 1.0, 0.5);
    vec3 c4 = vec3(1.0, 1.0, 0.0);
    vec3 c5 = vec3(1.0, 0.5, 0.0);
    vec3 c6 = vec3(1.0, 0.0, 0.0);
    
    if (value < 0.2) {
        return mix(c1, c2, value * 5.0);
    } else if (value < 0.4) {
        return mix(c2, c3, (value - 0.2) * 5.0);
    } else if (value < 0.6) {
        return mix(c3, c4, (value - 0.4) * 5.0);
    } else if (value < 0.8) {
        return mix(c4, c5, (value - 0.6) * 5.0);
    } else {
        return mix(c5, c6, (value - 0.8) * 5.0);
    }
}

void main() {
    float value = texture(texSampler, fragTexCoord).r;
    vec3 color = colormap(value);
    outColor = vec4(color, 1.0);
}
