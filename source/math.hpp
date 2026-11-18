#pragma once

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	constexpr Vec3 operator+(const Vec3& rhs) const {
		return { x + rhs.x, y + rhs.y, z + rhs.z };
	}

	constexpr Vec3 operator-(const Vec3& rhs) const {
		return { x - rhs.x, y - rhs.y, z - rhs.z };
	}

	constexpr Vec3 operator*(float scalar) const {
		return { x * scalar, y * scalar, z * scalar };
	}
};

struct Vec4 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;
};

inline float dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline Vec3 normalize(const Vec3& v) {
	const float len = std::sqrt(dot(v, v));
	if (len <= 1e-6f) {
		return { 0.0f, 0.0f, 0.0f };
	}
	return { v.x / len, v.y / len, v.z / len };
}

// Column-major 4x4 matrix (m[col][row]) compatible with GLSL mat4.
struct Mat4 {
	float m[4][4] = {};

	static constexpr Mat4 identity() {
		Mat4 res{};
		res.m[0][0] = 1.0f;
		res.m[1][1] = 1.0f;
		res.m[2][2] = 1.0f;
		res.m[3][3] = 1.0f;
		return res;
	}
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 res{};
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			res.m[col][row] =
				a.m[0][row] * b.m[col][0] +
				a.m[1][row] * b.m[col][1] +
				a.m[2][row] * b.m[col][2] +
				a.m[3][row] * b.m[col][3];
		}
	}
	return res;
}

inline Mat4 translation(const Vec3& t) {
	Mat4 res = Mat4::identity();
	res.m[3][0] = t.x;
	res.m[3][1] = t.y;
	res.m[3][2] = t.z;
	return res;
}

inline Mat4 scale(const Vec3& s) {
	Mat4 res{};
	res.m[0][0] = s.x;
	res.m[1][1] = s.y;
	res.m[2][2] = s.z;
	res.m[3][3] = 1.0f;
	return res;
}

inline Mat4 rotationX(float angle_rad) {
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	Mat4 res = Mat4::identity();
	res.m[1][1] = c;
	res.m[1][2] = s;
	res.m[2][1] = -s;
	res.m[2][2] = c;
	return res;
}

inline Mat4 rotationY(float angle_rad) {
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	Mat4 res = Mat4::identity();
	res.m[0][0] = c;
	res.m[0][2] = -s;
	res.m[2][0] = s;
	res.m[2][2] = c;
	return res;
}

inline Mat4 rotationZ(float angle_rad) {
	const float c = std::cos(angle_rad);
	const float s = std::sin(angle_rad);
	Mat4 res = Mat4::identity();
	res.m[0][0] = c;
	res.m[0][1] = s;
	res.m[1][0] = -s;
	res.m[1][1] = c;
	return res;
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
	const Vec3 f = normalize(target - eye);
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 res = Mat4::identity();
	res.m[0][0] = s.x;
	res.m[1][0] = s.y;
	res.m[2][0] = s.z;
	res.m[3][0] = -dot(s, eye);

	res.m[0][1] = u.x;
	res.m[1][1] = u.y;
	res.m[2][1] = u.z;
	res.m[3][1] = -dot(u, eye);

	res.m[0][2] = -f.x;
	res.m[1][2] = -f.y;
	res.m[2][2] = -f.z;
	res.m[3][2] = dot(f, eye);

	return res;
}

// Right-handed perspective projection for Vulkan clip space (Y down, Z in [0, 1]).
inline Mat4 perspectiveVulkan(float fov_y_rad, float aspect, float z_near, float z_far) {
	const float tan_half_fov = std::tan(fov_y_rad * 0.5f);
	Mat4 res{};
	res.m[0][0] = 1.0f / (aspect * tan_half_fov);
	res.m[1][1] = -1.0f / tan_half_fov;
	res.m[2][2] = z_far / (z_near - z_far);
	res.m[2][3] = -1.0f;
	res.m[3][2] = (z_far * z_near) / (z_near - z_far);
	return res;
}

// Right-handed orthographic projection for Vulkan clip space (Y down, Z in [0, 1]).
// z_near and z_far are positive distances along the camera's forward (-Z) direction.
inline Mat4 orthographicVulkan(float left, float right, float bottom, float top,
                               float z_near, float z_far) {
	Mat4 res{};
	res.m[0][0] = 2.0f / (right - left);
	res.m[1][1] = -2.0f / (top - bottom);
	res.m[2][2] = -1.0f / (z_far - z_near);
	res.m[3][0] = -(right + left) / (right - left);
	res.m[3][1] = (top + bottom) / (top - bottom);
	res.m[3][2] = -z_near / (z_far - z_near);
	res.m[3][3] = 1.0f;
	return res;
}

} // namespace math
