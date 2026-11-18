#include "application.hpp"
#include "math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

struct Vertex {
	math::Vec3 position;
	math::Vec3 color;
};

struct UniformBufferObject {
	math::Mat4 model;
	math::Mat4 view;
	math::Mat4 projection;
	math::Vec4 color_tint;
};

enum class ProjectionMode : int {
	Perspective = 0,
	Orthographic = 1,
};

enum class TrajectoryType : int {
	FigureEight = 0,
	Epitrochoid = 1,
	TrefoilKnot = 2,
};

struct SceneState {
	// Task 1: Projection parameters
	ProjectionMode projection_mode = ProjectionMode::Perspective;
	float perspective_fov_deg = 45.0f;
	float ortho_half_height = 2.5f;
	float camera_distance = 5.5f;

	// Task 2: Manual affine transforms
	float position[3] = { 0.0f, 0.0f, 0.0f };
	float rotation_deg[3] = { 20.0f, 35.0f, 0.0f };
	float scale[3] = { 1.0f, 1.0f, 1.0f };

	// Task 3: Complex trajectory animation
	bool trajectory_enabled = true;
	bool anim_playing = true;
	TrajectoryType trajectory_type = TrajectoryType::FigureEight;
	float anim_speed = 1.0f;
	float trajectory_radius = 1.4f;
	float vertical_amplitude = 0.65f;
	float frequency_ratio = 2.0f;
	float spin_speed_deg = 50.0f;
	float anim_time = 0.0f;
	double last_frame_time = 0.0;
	bool has_last_frame_time = false;

	// Task 4 & 5: Color multiplier for procedural vertex colors
	float color_tint[3] = { 1.0f, 1.0f, 1.0f };
};

SceneState state{};

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;

VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;
uint32_t index_count = 0;

VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_allocation = VK_NULL_HANDLE;
void* uniform_mapped_data = nullptr;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;

math::Vec3 computeProceduralColor(const math::Vec3& local_pos) {
	// Procedurally assign RGB from the vertex position in local cube coordinates [-0.5, 0.5]^3.
	const auto mapChannel = [](float coord) {
		return 0.15f + 0.85f * (coord + 0.5f);
	};
	return {
		mapChannel(local_pos.x),
		mapChannel(local_pos.y),
		mapChannel(local_pos.z),
	};
}

std::vector<uint32_t> readShaderSpirv(const char* shader_name) {
	const std::array<std::filesystem::path, 3> candidates = {
		std::filesystem::path("shaders") / shader_name,
		std::filesystem::path("..") / "shaders" / shader_name,
		std::filesystem::path("..") / ".." / "shaders" / shader_name,
	};

	for (const auto& path : candidates) {
		std::ifstream file(path, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			continue;
		}

		const std::streamsize size = file.tellg();
		if (size <= 0 || (size % 4) != 0) {
			std::cerr << "Invalid SPIR-V shader file size: " << path.string() << '\n';
			return {};
		}

		std::vector<uint32_t> buffer(static_cast<size_t>(size) / sizeof(uint32_t));
		file.seekg(0, std::ios::beg);
		if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
			std::cerr << "Failed to read SPIR-V shader file: " << path.string() << '\n';
			return {};
		}
		return buffer;
	}

	std::cerr << "Could not locate compiled shader: shaders/" << shader_name << '\n';
	return {};
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
	const VkShaderModuleCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module\n";
		return VK_NULL_HANDLE;
	}
	return shader_module;
}

bool createHostBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkBuffer& out_buffer, VmaAllocation& out_allocation, void** out_mapped = nullptr) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result_info{};
	if (vmaCreateBuffer(allocator, &buffer_info, &alloc_info,
	                    &out_buffer, &out_allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan buffer via VMA\n";
		return false;
	}

	if (out_mapped != nullptr) {
		*out_mapped = result_info.pMappedData;
	}
	return true;
}

bool createCubeGeometry() {
	auto& context = graphics::internal::context;

	const std::array<math::Vec3, 8> positions = {
		math::Vec3{ -0.5f, -0.5f,  0.5f }, // 0: Front-Bottom-Left
		math::Vec3{  0.5f, -0.5f,  0.5f }, // 1: Front-Bottom-Right
		math::Vec3{  0.5f,  0.5f,  0.5f }, // 2: Front-Top-Right
		math::Vec3{ -0.5f,  0.5f,  0.5f }, // 3: Front-Top-Left
		math::Vec3{ -0.5f, -0.5f, -0.5f }, // 4: Back-Bottom-Left
		math::Vec3{  0.5f, -0.5f, -0.5f }, // 5: Back-Bottom-Right
		math::Vec3{  0.5f,  0.5f, -0.5f }, // 6: Back-Top-Right
		math::Vec3{ -0.5f,  0.5f, -0.5f }, // 7: Back-Top-Left
	};

	std::array<Vertex, 8> vertices{};
	for (size_t i = 0; i < positions.size(); ++i) {
		vertices[i] = {
			.position = positions[i],
			.color = computeProceduralColor(positions[i]),
		};
	}

	// Outward counter-clockwise triangles for all 6 faces of the cube.
	const std::array<uint16_t, 36> indices = {
		0, 1, 2,  0, 2, 3, // Front (+Z)
		5, 4, 7,  5, 7, 6, // Back (-Z)
		1, 5, 6,  1, 6, 2, // Right (+X)
		4, 0, 3,  4, 3, 7, // Left (-X)
		3, 2, 6,  3, 6, 7, // Top (+Y)
		4, 5, 1,  4, 1, 0, // Bottom (-Y)
	};
	index_count = static_cast<uint32_t>(indices.size());

	void* mapped_vertices = nullptr;
	const VkDeviceSize vertex_bytes = sizeof(vertices);
	if (!createHostBuffer(context.allocator, vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                      vertex_buffer, vertex_allocation, &mapped_vertices)) {
		return false;
	}
	std::memcpy(mapped_vertices, vertices.data(), static_cast<size_t>(vertex_bytes));
	vmaFlushAllocation(context.allocator, vertex_allocation, 0, vertex_bytes);

	void* mapped_indices = nullptr;
	const VkDeviceSize index_bytes = sizeof(indices);
	if (!createHostBuffer(context.allocator, index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                      index_buffer, index_allocation, &mapped_indices)) {
		return false;
	}
	std::memcpy(mapped_indices, indices.data(), static_cast<size_t>(index_bytes));
	vmaFlushAllocation(context.allocator, index_allocation, 0, index_bytes);

	return true;
}

bool createUniformResources() {
	auto& context = graphics::internal::context;

	if (!createHostBuffer(context.allocator, sizeof(UniformBufferObject),
	                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                      uniform_buffer, uniform_allocation, &uniform_mapped_data)) {
		return false;
	}

	const VkDescriptorSetLayoutBinding ubo_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = nullptr,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &ubo_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	const VkDescriptorBufferInfo buffer_info = {
		.buffer = uniform_buffer,
		.offset = 0,
		.range = sizeof(UniformBufferObject),
	};

	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);
	return true;
}

bool createGraphicsPipeline() {
	auto& context = graphics::internal::context;

	const std::vector<uint32_t> vert_code = readShaderSpirv("cube.vert.spv");
	const std::vector<uint32_t> frag_code = readShaderSpirv("cube.frag.spv");
	if (vert_code.empty() || frag_code.empty()) {
		return false;
	}

	VkShaderModule vert_module = createShaderModule(context.device, vert_code);
	VkShaderModule frag_module = createShaderModule(context.device, frag_code);
	if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
		if (vert_module != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vert_module, nullptr);
		}
		if (frag_module != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, frag_module, nullptr);
		}
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding_desc = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attribute_descs[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = static_cast<uint32_t>(offsetof(Vertex, position)),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = static_cast<uint32_t>(offsetof(Vertex, color)),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_desc,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attribute_descs,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(context.device, vert_module, nullptr);
		vkDestroyShaderModule(context.device, frag_module, nullptr);
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(
		context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);

	vkDestroyShaderModule(context.device, vert_module, nullptr);
	vkDestroyShaderModule(context.device, frag_module, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

math::Vec3 evaluateTrajectoryOffset(const SceneState& s) {
	if (!s.trajectory_enabled) {
		return { 0.0f, 0.0f, 0.0f };
	}

	const float t = s.anim_time;
	const float r = s.trajectory_radius;
	const float amp = s.vertical_amplitude;
	const float freq = s.frequency_ratio;

	switch (s.trajectory_type) {
	case TrajectoryType::FigureEight:
		return {
			r * std::sin(t),
			amp * std::sin(freq * t),
			r * std::sin(t) * std::cos(t),
		};

	case TrajectoryType::Epitrochoid: {
		const float r2 = r * 0.35f;
		return {
			r * std::cos(t) + r2 * std::cos(freq * t),
			amp * std::sin(freq * t),
			r * std::sin(t) + r2 * std::sin(freq * t),
		};
	}

	case TrajectoryType::TrefoilKnot:
		return {
			0.45f * r * (std::sin(t) + 2.0f * std::sin(2.0f * t)),
			amp * std::sin(3.0f * t),
			0.45f * r * (std::cos(t) - 2.0f * std::cos(2.0f * t)),
		};
	}

	return { 0.0f, 0.0f, 0.0f };
}

math::Vec3 evaluateTrajectoryRotationDeg(const SceneState& s) {
	if (!s.trajectory_enabled) {
		return { 0.0f, 0.0f, 0.0f };
	}

	const float base_deg = s.anim_time * s.spin_speed_deg;
	return {
		base_deg * 0.7f,
		base_deg,
		base_deg * 0.4f,
	};
}

} // namespace

bool initialize() {
	if (!createCubeGeometry()) {
		shutdown();
		return false;
	}
	if (!createUniformResources()) {
		shutdown();
		return false;
	}
	if (!createGraphicsPipeline()) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		graphics_pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		descriptor_set = VK_NULL_HANDLE;
	}
	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
		descriptor_set_layout = VK_NULL_HANDLE;
	}
	if (uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, uniform_buffer, uniform_allocation);
		uniform_buffer = VK_NULL_HANDLE;
		uniform_allocation = VK_NULL_HANDLE;
		uniform_mapped_data = nullptr;
	}
	if (index_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
		index_buffer = VK_NULL_HANDLE;
		index_allocation = VK_NULL_HANDLE;
	}
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_allocation = VK_NULL_HANDLE;
	}
}

void update(double time) {
	if (!state.has_last_frame_time) {
		state.last_frame_time = time;
		state.has_last_frame_time = true;
	}

	const double raw_dt = time - state.last_frame_time;
	state.last_frame_time = time;
	const float dt = static_cast<float>(std::clamp(raw_dt, 0.0, 0.1));

	if (state.trajectory_enabled && state.anim_playing) {
		state.anim_time += dt * state.anim_speed;
	}

	ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 560.0f), ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Lab 1: 3D Cube (Variant 1)")) {
		// 1. Projection controls
		if (ImGui::CollapsingHeader("1. Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
			int mode = static_cast<int>(state.projection_mode);
			ImGui::RadioButton("Perspective", &mode, static_cast<int>(ProjectionMode::Perspective));
			ImGui::SameLine();
			ImGui::RadioButton("Orthographic", &mode, static_cast<int>(ProjectionMode::Orthographic));
			state.projection_mode = static_cast<ProjectionMode>(mode);

			if (state.projection_mode == ProjectionMode::Perspective) {
				ImGui::SliderFloat("FOV (deg)", &state.perspective_fov_deg, 15.0f, 120.0f, "%.1f");
			} else {
				ImGui::SliderFloat("Ortho Half-Height", &state.ortho_half_height, 0.5f, 8.0f, "%.2f");
			}
			ImGui::SliderFloat("Camera Distance", &state.camera_distance, 2.0f, 15.0f, "%.2f");
		}

		// 2. Transform controls
		if (ImGui::CollapsingHeader("2. Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", state.position, 0.02f, -5.0f, 5.0f, "%.2f");
			ImGui::DragFloat3("Rotation (deg)", state.rotation_deg, 0.5f, -360.0f, 360.0f, "%.1f");
			ImGui::DragFloat3("Scale", state.scale, 0.02f, 0.1f, 4.0f, "%.2f");
			for (float& axis_scale : state.scale) {
				axis_scale = std::max(axis_scale, 0.05f);
			}

			if (ImGui::Button("Reset Transform")) {
				state.position[0] = state.position[1] = state.position[2] = 0.0f;
				state.rotation_deg[0] = 20.0f;
				state.rotation_deg[1] = 35.0f;
				state.rotation_deg[2] = 0.0f;
				state.scale[0] = state.scale[1] = state.scale[2] = 1.0f;
			}
		}

		// 3. Trajectory animation controls
		if (ImGui::CollapsingHeader("3. Trajectory Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Enable Trajectory", &state.trajectory_enabled);

			if (ImGui::Button(state.anim_playing ? "Pause" : "Play", ImVec2(90.0f, 0.0f))) {
				state.anim_playing = !state.anim_playing;
			}
			ImGui::SameLine();
			if (ImGui::Button("Reset Animation")) {
				state.anim_time = 0.0f;
			}

			const char* trajectory_names[] = {
				"Figure-8 (Lissajous 3D)",
				"Epitrochoid Orbit",
				"Trefoil Knot",
			};
			int traj_idx = static_cast<int>(state.trajectory_type);
			if (ImGui::Combo("Trajectory", &traj_idx, trajectory_names, 3)) {
				state.trajectory_type = static_cast<TrajectoryType>(traj_idx);
			}

			ImGui::SliderFloat("Animation Speed", &state.anim_speed, -3.0f, 3.0f, "%.2f");
			ImGui::SliderFloat("Trajectory Radius", &state.trajectory_radius, 0.0f, 3.0f, "%.2f");
			ImGui::SliderFloat("Vertical Amplitude", &state.vertical_amplitude, 0.0f, 2.0f, "%.2f");
			ImGui::SliderFloat("Frequency Ratio", &state.frequency_ratio, 0.5f, 5.0f, "%.2f");
			ImGui::SliderFloat("Spin Speed (deg/s)", &state.spin_speed_deg, 0.0f, 180.0f, "%.1f");

			const math::Vec3 offset = evaluateTrajectoryOffset(state);
			ImGui::Text("t = %.2f | offset = (%.2f, %.2f, %.2f)",
			            state.anim_time, offset.x, offset.y, offset.z);
		}

		// 4 & 5. Color tint multiplying procedural vertex colors
		if (ImGui::CollapsingHeader("4. Color (Procedural x Tint)", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Color Tint", state.color_tint);
			if (ImGui::Button("Reset Color (White)")) {
				state.color_tint[0] = 1.0f;
				state.color_tint[1] = 1.0f;
				state.color_tint[2] = 1.0f;
			}
			ImGui::TextWrapped(
				"Vertex colors are procedurally assigned from local XYZ positions "
				"and multiplied by Color Tint in the fragment shader.");
		}
	}
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (fd.command_buffer == VK_NULL_HANDLE ||
	    context.swapchain_extent.width == 0 ||
	    context.swapchain_extent.height == 0) {
		return;
	}

	const math::Vec3 traj_offset = evaluateTrajectoryOffset(state);
	const math::Vec3 traj_rot_deg = evaluateTrajectoryRotationDeg(state);

	const math::Vec3 final_pos = {
		state.position[0] + traj_offset.x,
		state.position[1] + traj_offset.y,
		state.position[2] + traj_offset.z,
	};
	const math::Vec3 final_rot_rad = {
		math::radians(state.rotation_deg[0] + traj_rot_deg.x),
		math::radians(state.rotation_deg[1] + traj_rot_deg.y),
		math::radians(state.rotation_deg[2] + traj_rot_deg.z),
	};
	const math::Vec3 final_scale = {
		state.scale[0],
		state.scale[1],
		state.scale[2],
	};

	const math::Mat4 model =
		math::translation(final_pos) *
		math::rotationZ(final_rot_rad.z) *
		math::rotationY(final_rot_rad.y) *
		math::rotationX(final_rot_rad.x) *
		math::scale(final_scale);

	const math::Vec3 camera_eye = { 0.0f, 1.2f, state.camera_distance };
	const math::Vec3 camera_target = { 0.0f, 0.0f, 0.0f };
	const math::Vec3 camera_up = { 0.0f, 1.0f, 0.0f };
	const math::Mat4 view = math::lookAt(camera_eye, camera_target, camera_up);

	const float aspect = static_cast<float>(context.swapchain_extent.width) /
	                     static_cast<float>(context.swapchain_extent.height);

	math::Mat4 projection{};
	if (state.projection_mode == ProjectionMode::Perspective) {
		projection = math::perspectiveVulkan(
			math::radians(state.perspective_fov_deg), aspect, 0.1f, 100.0f);
	} else {
		const float half_h = state.ortho_half_height;
		const float half_w = half_h * aspect;
		projection = math::orthographicVulkan(
			-half_w, half_w, -half_h, half_h, 0.1f, 100.0f);
	}

	const UniformBufferObject ubo = {
		.model = model,
		.view = view,
		.projection = projection,
		.color_tint = { state.color_tint[0], state.color_tint[1], state.color_tint[2], 1.0f },
	};

	std::memcpy(uniform_mapped_data, &ubo, sizeof(ubo));
	vmaFlushAllocation(context.allocator, uniform_allocation, 0, sizeof(ubo));

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.1f, 0.1f, 0.14f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &vertex_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
