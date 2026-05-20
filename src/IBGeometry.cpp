/**
 * IBGeometry.cpp
 *
 * Host-side loaders for the Lagrangian geometry used by the future
 * Taira-Colonius immersed-boundary projection block.
 */

#include "IBGeometry.H"

#include <AMReX.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ibm3d {
namespace {

using Point = IBGeometry::Point;
using Element = IBGeometry::Element;

void AbortGeometryRead(const std::string &path, const std::string &message) {
  amrex::Abort("IB geometry '" + path + "': " + message);
}

void CheckFinite(const std::string &path, const Point &point) {
  for (const auto coord : point) {
    if (!std::isfinite(coord))
      AbortGeometryRead(path, "point coordinates must be finite");
  }
}

void CheckElement(const std::string &path, const Element &element,
                  int npoints) {
  for (const auto index : element) {
    if (index < 0 || index >= npoints)
      AbortGeometryRead(path, "connectivity index is out of range");
  }
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    for (int j = i + 1; j < AMREX_SPACEDIM; ++j) {
      if (element[i] == element[j])
        AbortGeometryRead(path, "degenerate element has repeated vertices");
    }
  }
}

void AddElement(IBGeometry &geometry, const Element &element,
                const std::string &path) {
  CheckElement(path, element, static_cast<int>(geometry.points.size()));
  geometry.elements.push_back(element);
}

#if (AMREX_SPACEDIM == 2)

void CheckPositiveCount(const std::string &path, long long value,
                        const char *name) {
  if (value <= 0)
    AbortGeometryRead(path, std::string(name) + " must be positive");
  if (value > static_cast<long long>(std::numeric_limits<int>::max()))
    AbortGeometryRead(path, std::string(name) + " exceeds int range");
}

#elif (AMREX_SPACEDIM == 3)

int AddPoint(IBGeometry &geometry, std::map<Point, int> &point_index,
             const Point &point, const std::string &path) {
  CheckFinite(path, point);
  if (const auto it = point_index.find(point); it != point_index.end())
    return it->second;

  if (geometry.points.size() >=
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    AbortGeometryRead(path, "too many unique points");

  const auto index = static_cast<int>(geometry.points.size());
  geometry.points.push_back(point);
  point_index.emplace(point, index);
  return index;
}

std::uint32_t DecodeUInt32LE(const unsigned char *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

float DecodeFloat32LE(const unsigned char *bytes) {
  const auto bits = DecodeUInt32LE(bytes);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

#endif

#if (AMREX_SPACEDIM == 2)

int InferConnectivityBase(
    const std::string &path,
    const std::vector<std::array<long long, AMREX_SPACEDIM>> &raw_elements,
    int npoints) {
  auto valid_with_base = [&](int base) {
    return std::all_of(
        raw_elements.begin(), raw_elements.end(), [&](const auto &element) {
          return std::all_of(element.begin(), element.end(),
                             [&](long long raw_index) {
                               const auto index = raw_index - base;
                               return index >= 0 && index < npoints;
                             });
        });
  };

  if (valid_with_base(0))
    return 0;
  if (valid_with_base(1))
    return 1;

  AbortGeometryRead(path, "connectivity is neither zero-based nor one-based");
  return 0; // unreachable after amrex::Abort
}

#elif (AMREX_SPACEDIM == 3)

bool ReadBinaryTriangleCount(const std::string &path,
                             std::uint32_t &triangle_count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    AbortGeometryRead(path, "could not open file");

  const auto file_size = input.tellg();
  if (file_size < static_cast<std::streamoff>(84))
    return false;

  input.seekg(80);
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    AbortGeometryRead(path, "could not read binary STL triangle count");

  triangle_count = DecodeUInt32LE(bytes.data());
  const auto expected_size = static_cast<std::streamoff>(84) +
                             static_cast<std::streamoff>(triangle_count) *
                                 static_cast<std::streamoff>(50);
  return file_size == expected_size;
}

void AddSTLTriangle(IBGeometry &geometry, std::map<Point, int> &point_index,
                    const std::array<Point, 3> &vertices,
                    const std::string &path) {
  Element triangle{};
  for (int d = 0; d < 3; ++d)
    triangle[d] = AddPoint(geometry, point_index, vertices[d], path);
  AddElement(geometry, triangle, path);
}

IBGeometry LoadBinarySTL(const std::string &path,
                         std::uint32_t triangle_count) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    AbortGeometryRead(path, "could not open file");

  std::array<char, 80> header{};
  input.read(header.data(), static_cast<std::streamsize>(header.size()));

  std::array<unsigned char, 4> count_bytes{};
  input.read(reinterpret_cast<char *>(count_bytes.data()),
             static_cast<std::streamsize>(count_bytes.size()));
  if (!input)
    AbortGeometryRead(path, "could not read binary STL header");

  IBGeometry geometry;
  std::map<Point, int> point_index;

  std::array<unsigned char, 50> record{};
  for (std::uint32_t tri = 0; tri < triangle_count; ++tri) {
    input.read(reinterpret_cast<char *>(record.data()),
               static_cast<std::streamsize>(record.size()));
    if (!input)
      AbortGeometryRead(path, "unexpected EOF in binary STL triangle record");

    std::array<Point, 3> vertices{};
    for (int vertex = 0; vertex < 3; ++vertex) {
      for (int d = 0; d < 3; ++d) {
        const auto offset = 12 + vertex * 12 + d * 4;
        vertices[vertex][d] =
            static_cast<amrex::Real>(DecodeFloat32LE(record.data() + offset));
      }
    }
    AddSTLTriangle(geometry, point_index, vertices, path);
  }

  if (geometry.elements.empty())
    AbortGeometryRead(path, "STL contains no triangles");
  return geometry;
}

std::string LowercaseAscii(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  return token;
}

IBGeometry LoadAsciiSTL(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    AbortGeometryRead(path, "could not open file");

  IBGeometry geometry;
  std::map<Point, int> point_index;

  std::array<Point, 3> vertices{};
  int vertex_count = 0;

  std::string token;
  while (input >> token) {
    if (LowercaseAscii(token) != "vertex")
      continue;

    Point point{};
    for (int d = 0; d < 3; ++d) {
      if (!(input >> point[d]))
        AbortGeometryRead(path, "malformed ASCII STL vertex");
    }

    vertices[vertex_count++] = point;
    if (vertex_count == 3) {
      AddSTLTriangle(geometry, point_index, vertices, path);
      vertex_count = 0;
    }
  }

  if (vertex_count != 0)
    AbortGeometryRead(path, "ASCII STL vertex count is not a multiple of 3");
  if (geometry.elements.empty())
    AbortGeometryRead(path, "STL contains no triangles");
  return geometry;
}

#endif

} // namespace

#if (AMREX_SPACEDIM == 2)

IBGeometry LoadIBCurveAscii(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    AbortGeometryRead(path, "could not open file");

  long long npoints_raw = 0;
  long long nelements_raw = 0;
  if (!(input >> npoints_raw >> nelements_raw))
    AbortGeometryRead(path, "first line must contain n_points n_segments");

  CheckPositiveCount(path, npoints_raw, "n_points");
  CheckPositiveCount(path, nelements_raw, "n_segments");

  const auto npoints = static_cast<int>(npoints_raw);
  const auto nelements = static_cast<int>(nelements_raw);

  IBGeometry geometry;
  geometry.points.resize(static_cast<std::size_t>(npoints));
  for (auto &point : geometry.points) {
    for (auto &coord : point) {
      if (!(input >> coord))
        AbortGeometryRead(path, "not enough point coordinates");
    }
    CheckFinite(path, point);
  }

  std::vector<std::array<long long, AMREX_SPACEDIM>> raw_elements(
      static_cast<std::size_t>(nelements));
  for (auto &element : raw_elements) {
    for (auto &index : element) {
      if (!(input >> index))
        AbortGeometryRead(path, "not enough connectivity entries");
    }
  }

  const auto base = InferConnectivityBase(path, raw_elements, npoints);
  geometry.elements.reserve(static_cast<std::size_t>(nelements));
  for (const auto &raw_element : raw_elements) {
    Element element{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      element[d] = static_cast<int>(raw_element[d] - base);
    AddElement(geometry, element, path);
  }

  std::string trailing;
  if (input >> trailing)
    AbortGeometryRead(path, "unexpected trailing token '" + trailing + "'");

  return geometry;
}

#elif (AMREX_SPACEDIM == 3)

IBGeometry LoadIBSurfaceSTL(const std::string &path) {
  std::uint32_t triangle_count = 0;
  if (ReadBinaryTriangleCount(path, triangle_count))
    return LoadBinarySTL(path, triangle_count);
  return LoadAsciiSTL(path);
}

#endif

IBGeometry LoadIBGeometry(const std::string &path) {
#if (AMREX_SPACEDIM == 2)
  return LoadIBCurveAscii(path);
#elif (AMREX_SPACEDIM == 3)
  return LoadIBSurfaceSTL(path);
#endif
}

} // namespace ibm3d
