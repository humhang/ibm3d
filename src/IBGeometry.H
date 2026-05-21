#ifndef IB_GEOMETRY_H
#define IB_GEOMETRY_H

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace ibm3d {

struct IBMarker {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> x{};
  amrex::Real weight = 0.0;
};

struct IBGeometry {
  using Point = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;
  using Element = amrex::GpuArray<int, AMREX_SPACEDIM>;

  std::vector<Point> points;
  std::vector<Element> elements;
  std::vector<IBMarker> markers;

  amrex::Gpu::DeviceVector<Point> device_points;
  amrex::Gpu::DeviceVector<Element> device_elements;
  amrex::Gpu::DeviceVector<IBMarker> device_markers;

  void BuildMarkers(const std::string &source_path);
  void UploadToDevice();
  void ClearDevice();
};

#if (AMREX_SPACEDIM == 2)
/// Load an ASCII immersed curve:
///   n_points n_segments
///   x y
///   ...
///   i j
///   ...
/// Connectivity may be zero-based or one-based; it is stored zero-based.
IBGeometry LoadIBCurveAscii(const std::string &path);
#elif (AMREX_SPACEDIM == 3)
/// Load an ASCII or binary STL immersed surface.  STL stores triangle
/// vertices, not indexed topology; repeated coordinates are deduplicated.
IBGeometry LoadIBSurfaceSTL(const std::string &path);
#else
#error "IBGeometry supports only AMREX_SPACEDIM == 2 or 3"
#endif

/// Dimension-selected front end: 2D uses the ASCII curve format, 3D uses STL.
IBGeometry LoadIBGeometry(const std::string &path);

} // namespace ibm3d

#endif
