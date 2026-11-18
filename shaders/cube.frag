#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 projection;
	vec4 color_tint;
} ubo;

layout(location = 0) in vec3 frag_color;

layout(location = 0) out vec4 out_color;

void main() {
	out_color = vec4(frag_color * ubo.color_tint.rgb, 1.0);
}
