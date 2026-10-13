import bpy
import json
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Your laptop ID inside the JSON file
LAPTOP_ID = "dell_laptop"

# Path to your JSON file
JSON_PATH = Path(r"C:\Users\panosvrachnos\Documents\thesis_env\controllers\laptop_database.json")

# Output folder where OBJ and MTL files will be written
# Example: C:\Users\YourName\my_webots_project\protos
OUT_DIR = Path(r"C:\Users\panosvrachnos\Desktop\Thesis\simulation data\handmade\handmade_laptop_variations")

# Base OBJ name
BASE_OBJ_NAME = "handmade_laptop.obj"

# Prefix for the generated variation OBJ files
OBJ_PREFIX = "handmade_laptop"

# Shared MTL file name
COMMON_MTL_NAME_IN_OBJ = "handmade_laptop.mtl"

# Path style written into the updated JSON
JSON_OBJ_PREFIX = "../protos"

# Collections that contain geometry to export
# Lights and Materials are intentionally excluded.
EXPORT_COLLECTIONS = [
    "Collection",
    "Buttons",
    "Texts",
]

# Exact Blender object names for the four screws
SCREW_OBJECTS = {
    "left_down": "left_down_screw",
    "left_up": "left_up_screw",
    "right_down": "right_down_screw",
    "right_up": "right_up_screw",
}

# If Texts contains Blender FONT objects, temporary mesh copies will be created.
# The original text objects are not destroyed.
CONVERT_TEXT_OBJECTS_TO_MESH = True

# Write a new JSON file with all 16 OBJ paths
WRITE_UPDATED_JSON = True

# ============================================================
# INTERNAL HELPERS
# ============================================================

TEMP_COLLECTION_NAME = "_TEMP_EXPORT_TEXT_MESHES"


def get_or_create_temp_collection():
    temp_collection = bpy.data.collections.get(TEMP_COLLECTION_NAME)

    if temp_collection is None:
        temp_collection = bpy.data.collections.new(TEMP_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(temp_collection)

    return temp_collection


def clear_temp_collection():
    temp_collection = bpy.data.collections.get(TEMP_COLLECTION_NAME)

    if temp_collection is None:
        return

    for obj in list(temp_collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.data.collections.remove(temp_collection)


def make_mesh_copy_from_text_object(obj, temp_collection):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)

    mesh = bpy.data.meshes.new_from_object(evaluated_obj, depsgraph=depsgraph)
    mesh.name = f"{obj.name}_export_mesh"

    mesh_obj = bpy.data.objects.new(f"{obj.name}_export_mesh", mesh)
    mesh_obj.matrix_world = obj.matrix_world.copy()

    # Preserve materials from the original text object
    if hasattr(obj.data, "materials"):
        for material in obj.data.materials:
            mesh.materials.append(material)

    temp_collection.objects.link(mesh_obj)

    return mesh_obj


def get_export_objects():
    """
    Collects all exportable objects from EXPORT_COLLECTIONS.

    - MESH objects are used directly.
    - FONT objects are converted to temporary mesh copies if enabled.
    - Cameras, lights, and other non-geometry objects are ignored.
    """

    objects_by_name = {}

    temp_collection = None

    if CONVERT_TEXT_OBJECTS_TO_MESH:
        clear_temp_collection()
        temp_collection = get_or_create_temp_collection()

    for collection_name in EXPORT_COLLECTIONS:
        collection = bpy.data.collections.get(collection_name)

        if collection is None:
            raise RuntimeError(f"Collection not found: {collection_name}")

        for obj in collection.all_objects:
            if obj.type == "MESH":
                objects_by_name[obj.name] = obj

            elif obj.type == "FONT" and CONVERT_TEXT_OBJECTS_TO_MESH:
                mesh_copy = make_mesh_copy_from_text_object(obj, temp_collection)
                objects_by_name[mesh_copy.name] = mesh_copy

    return list(objects_by_name.values())


def patch_obj_mtllib(obj_path: Path, mtllib_name: str):
    """
    Forces the OBJ file to use one shared MTL file.

    Example line written inside every OBJ:
        mtllib laptop_dell.mtl
    """

    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    new_lines = []
    mtllib_written = False

    for line in lines:
        if line.startswith("mtllib "):
            if not mtllib_written:
                new_lines.append(f"mtllib {mtllib_name}")
                mtllib_written = True
            continue

        new_lines.append(line)

    if not mtllib_written:
        insert_at = 0

        while insert_at < len(new_lines) and new_lines[insert_at].startswith("#"):
            insert_at += 1

        new_lines.insert(insert_at, f"mtllib {mtllib_name}")

    obj_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def select_objects_for_version(export_objects, version_classes):
    """
    Selects the correct objects for one version.

    Normal laptop objects:
        always exported

    Screw objects:
        exported only when JSON says "screw"
    """

    screw_name_to_key = {
        blender_object_name: json_key
        for json_key, blender_object_name in SCREW_OBJECTS.items()
    }

    bpy.ops.object.select_all(action="DESELECT")

    selected_objects = []

    for obj in export_objects:
        screw_key = screw_name_to_key.get(obj.name)

        if screw_key is None:
            should_export = True
        else:
            should_export = version_classes[screw_key] == "screw"

        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(should_export)

        if should_export:
            selected_objects.append(obj)

    if not selected_objects:
        raise RuntimeError("No objects selected for export.")

    bpy.context.view_layer.objects.active = selected_objects[0]


def export_obj(obj_path: Path):
    """
    Exports only the currently selected objects.
    """

    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        path_mode="RELATIVE",
    )


def export_one_variant(version_code, version_data, export_objects, common_mtl_path):
    """
    Exports one laptop variation, for example:
        laptop_dell_0000.obj
        laptop_dell_0001.obj
        ...
        laptop_dell_1111.obj
    """

    obj_path = OUT_DIR / f"{OBJ_PREFIX}_{version_code}.obj"

    select_objects_for_version(
        export_objects=export_objects,
        version_classes=version_data["classes"],
    )

    export_obj(obj_path)
    patch_obj_mtllib(obj_path, COMMON_MTL_NAME_IN_OBJ)

    generated_mtl = obj_path.with_suffix(".mtl")

    # When exporting 1111, use its generated MTL as the shared MTL
    # only if laptop_dell.mtl does not already exist.
    if version_code == "1111" and generated_mtl.exists() and not common_mtl_path.exists():
        shutil.move(str(generated_mtl), str(common_mtl_path))

    # Delete per-version MTL files, because all OBJs use laptop_dell.mtl.
    elif generated_mtl.exists() and generated_mtl.name != COMMON_MTL_NAME_IN_OBJ:
        generated_mtl.unlink()

    print(f"Exported: {obj_path.name}")


def copy_full_version_to_base_obj():
    """
    Creates:
        laptop_dell.obj

    by copying:
        laptop_dell_1111.obj

    So laptop_dell.obj is the full laptop with all screws.
    """

    source = OUT_DIR / f"{OBJ_PREFIX}_1111.obj"
    target = OUT_DIR / BASE_OBJ_NAME

    if not source.exists():
        raise RuntimeError(f"Cannot create base OBJ. Missing: {source}")

    shutil.copyfile(source, target)
    patch_obj_mtllib(target, COMMON_MTL_NAME_IN_OBJ)

    print(f"Created base OBJ: {target.name}")


def validate_screw_objects_exist():
    missing = []

    for json_key, blender_object_name in SCREW_OBJECTS.items():
        if bpy.data.objects.get(blender_object_name) is None:
            missing.append(blender_object_name)

    if missing:
        raise RuntimeError(
            "The following screw objects were not found in Blender:\n"
            + "\n".join(missing)
            + "\n\nCheck spelling, capitalization, and .001 suffixes."
        )
        
        
def patch_mtl_texture_paths(mtl_path: Path, texture_folder="textures"):
    """
    Forces texture paths inside the MTL to point to:

        textures/filename.png

    Example:
        map_Kd mat1.png
    becomes:
        map_Kd textures/mat1.png
    """

    if not mtl_path.exists():
        print(f"MTL not found, skipping texture patch: {mtl_path}")
        return

    texture_keywords = {
        "map_Ka",
        "map_Kd",
        "map_Ks",
        "map_Ke",
        "map_Ns",
        "map_d",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "norm",
        "refl",
    }

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        parts = stripped.split(maxsplit=1)

        if len(parts) != 2:
            new_lines.append(line)
            continue

        keyword, texture_path = parts

        if keyword not in texture_keywords:
            new_lines.append(line)
            continue

        texture_path = texture_path.replace("\\", "/")

        # Keep absolute paths unchanged
        if ":" in texture_path or texture_path.startswith("/"):
            new_lines.append(f"{keyword} {texture_path}")
            continue

        # If path already starts with textures/, keep it
        if texture_path.startswith(f"{texture_folder}/"):
            new_lines.append(f"{keyword} {texture_path}")
            continue

        # Otherwise use only the filename and force textures/
        filename = Path(texture_path).name.replace(" ", "_")
        new_lines.append(f"{keyword} {texture_folder}/{filename}")

    mtl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"Patched texture paths in: {mtl_path.name}")

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_screw_objects_exist()

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    versions = data["versions"]
    laptop = data["laptops"][LAPTOP_ID]

    common_mtl_path = OUT_DIR / COMMON_MTL_NAME_IN_OBJ

    export_objects = get_export_objects()

    print(f"Found {len(export_objects)} exportable objects.")
    print("Starting export...")

    # Export 1111 first so it can generate the shared material file.
    ordered_version_codes = ["1111"] + sorted(
        code for code in versions.keys()
        if code != "1111"
    )

    laptop["version_objs"] = {}

    for version_code in ordered_version_codes:
        export_one_variant(
            version_code=version_code,
            version_data=versions[version_code],
            export_objects=export_objects,
            common_mtl_path=common_mtl_path,
        )

        laptop["version_objs"][version_code] = (
            f"{JSON_OBJ_PREFIX}/{OBJ_PREFIX}_{version_code}.obj"
        )

    copy_full_version_to_base_obj()
    
    patch_mtl_texture_paths(common_mtl_path, texture_folder="textures")

    # Make sure the JSON still points to the shared MTL file
    laptop["mtl_file"] = f"{JSON_OBJ_PREFIX}/{COMMON_MTL_NAME_IN_OBJ}"

    if WRITE_UPDATED_JSON:
        updated_json_path = JSON_PATH.with_name(
            JSON_PATH.stem + "_with_all_versions.json"
        )

        updated_json_path.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8",
        )

        print(f"Updated JSON written to: {updated_json_path}")

    clear_temp_collection()

    print("Done.")


main()