#!/usr/bin/env python3
"""Download 2018 AlphaEarth Foundations embeddings for soil sampling points."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely import force_2d
from shapely.geometry import Point, mapping


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "黑土层采样数据" / "黑土层采样数据.csv"
ROI_SHP = ROOT / "data" / "songnen_plain" / "songnen_wgs84.shp"
OUT_DIR = ROOT / "alphaearth" / "data"
DATASET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_YEAR = 2018
TARGETS = ["pH值", "全碳(g/kg)", "有机碳(g/kg)", "容重(g/cm3)", "N(g/kg)"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.environ.get("EE_PROJECT"),
        help="Google Cloud project enabled for Earth Engine.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="AlphaEarth annual embedding year.",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=1000,
        help="Scale in meters for the low-resolution preview GeoTIFF.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Only download point samples, not the preview GeoTIFF.",
    )
    parser.add_argument(
        "--drive-export",
        action="store_true",
        help="Create a Google Drive table export task instead of downloading the CSV locally.",
    )
    return parser.parse_args()


def init_ee(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Provide a project with "
            "`--project YOUR_PROJECT_ID` or `EE_PROJECT=YOUR_PROJECT_ID`."
        ) from exc


def load_roi() -> tuple[gpd.GeoDataFrame, ee.Geometry]:
    roi = gpd.read_file(ROI_SHP).to_crs("EPSG:4326")
    roi["geometry"] = roi.geometry.map(force_2d)
    union = roi.geometry.union_all()
    return roi, ee.Geometry(mapping(union))


def load_samples_in_roi(roi: gpd.GeoDataFrame) -> pd.DataFrame:
    samples = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    samples = samples.dropna(subset=["经度", "纬度", *TARGETS]).copy()
    points = gpd.GeoDataFrame(
        samples,
        geometry=[Point(xy) for xy in zip(samples["经度"], samples["纬度"], strict=True)],
        crs="EPSG:4326",
    )
    roi_union = roi.geometry.union_all()
    points = points[points.geometry.within(roi_union)].copy()
    if points.empty:
        raise ValueError("No soil samples fall inside the Songnen Plain boundary.")
    return pd.DataFrame(points.drop(columns="geometry"))


def samples_to_ee(samples: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    keep_cols = ["样点", "点号", "经度", "纬度", *TARGETS]
    for row in samples[keep_cols].to_dict(orient="records"):
        lon = float(row["经度"])
        lat = float(row["纬度"])
        props = {key: value for key, value in row.items() if pd.notna(value)}
        features.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))
    return ee.FeatureCollection(features)


def build_sampled_feature_collection(samples: pd.DataFrame, year: int) -> ee.FeatureCollection:
    collection = (
        ee.ImageCollection(DATASET)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(EMBEDDING_BANDS)
    )

    def add_embeddings(feature: ee.Feature) -> ee.Feature:
        point = feature.geometry()
        image = collection.filterBounds(point).mosaic()
        values_info = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=10,
            maxPixels=1024,
        )
        return feature.set(values_info)

    return samples_to_ee(samples).map(add_embeddings)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    init_ee(args.project)
    roi_gdf, roi_ee = load_roi()
    samples = load_samples_in_roi(roi_gdf)

    preview_image = (
        ee.ImageCollection(DATASET)
        .filterBounds(roi_ee)
        .filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01")
        .mosaic()
        .select(EMBEDDING_BANDS)
    )

    sampled = build_sampled_feature_collection(samples, args.year)
    sample_out = OUT_DIR / f"alphaearth_{args.year}_sample_embeddings.csv"
    selectors = ["样点", "点号", "经度", "纬度", *TARGETS, *EMBEDDING_BANDS]

    if args.drive_export:
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f"alphaearth_{args.year}_sample_embeddings",
            fileNamePrefix=f"alphaearth_{args.year}_sample_embeddings",
            fileFormat="CSV",
            selectors=selectors,
        )
        task.start()
        print(f"Started Drive export task: {task.id}")
        print("Download the CSV from Google Drive, then place it at:")
        print(sample_out)
    else:
        geemap.ee_export_vector(
            sampled,
            filename=str(sample_out),
            selectors=selectors,
            timeout=600,
        )
        sampled_df = pd.read_csv(sample_out, encoding="utf-8")
        print(f"Wrote {len(sampled_df)} sampled rows: {sample_out}")

    if not args.skip_preview:
        preview = preview_image.select(["A00", "A01", "A02"]).clip(roi_ee)
        preview_out = OUT_DIR / f"alphaearth_{args.year}_preview_A00_A02.tif"
        geemap.ee_export_image(
            preview,
            filename=str(preview_out),
            scale=args.preview_scale,
            region=roi_ee,
            file_per_band=False,
        )
        print(f"Wrote preview GeoTIFF: {preview_out}")


if __name__ == "__main__":
    main()
