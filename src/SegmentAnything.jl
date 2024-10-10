module SegmentAnything

using Pkg
using Plots
using Images
using Colors

# declare python env
const path = @__DIR__
const parent_dir = dirname(path)
ENV["PYTHON"] = joinpath(parent_dir, "deps", "python", ".venv", "bin", "python")
Pkg.build("PyCall")
using PyCall


# load python libraries
const segment_anything = pyimport("segment_anything")
const cv2 = pyimport("cv2")
const plt = pyimport("matplotlib.pyplot")
const np = pyimport("numpy")
const torch = pyimport("torch")


"""
"""
function detect_device()
    
    if torch.cuda.is_available()
        println("CUDA is available")
        return "cuda"
    else
        println("CUDA is not available: CPU selected")
        return "cpu" 
    end
    
end

"""
"""
function load_sam_model(model_type::String)
    valid_models = Dict(
        "vit_b" => "sam_vit_b_01ec64.pth",
        "vit_l" => "sam_vit_l_0b3195.pth",
        "vit_h" => "sam_vit_h_4b8939.pth"
    )

    if haskey(valid_models, model_type)
        sam_checkpoint = joinpath(parent_dir, "deps", "sam", valid_models[model_type])
        println("Load SAM model: ", model_type, " ($(sam_checkpoint))")
        
        sam_model_registry = segment_anything.sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        return sam
    else
        error("Unknown model: $model_type. Please, select 'vit_b' or 'vit_l' or 'vit_h'.")
    end
end

"""
"""
function configure_mask_generator(sam, device)
    sam.to(device=device)
    
    mask_generator = segment_anything.SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.96,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return mask_generator
end


"""
    segment_image(image_path::String, model_type::String="vit_b")

Charge l'image à partir du chemin spécifié, effectue la segmentation avec le modèle choisi, et affiche les masques.

# Arguments
- `image_path`: Chemin vers l'image à segmenter.
- `model_type`: Type de modèle SAM à utiliser. Valeurs possibles : "vit_b", "vit_l", "vit_h". Par défaut : "vit_b".
"""
function segment_image(image_path::String, model_type::String="vit_b")
    device = detect_device()
    sam = load_sam_model(model_type)
    mask_generator = configure_mask_generator(sam, device)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)
    println("Nombre de masques générés : ", length(masks))

    return masks
end

"""
    show_anns_jl(anns, image)

Affiche les masques de segmentation en utilisant des fonctionnalités Julia.

# Arguments
- `anns`: Un tableau de dictionnaires contenant les masques générés par le modèle.
- `image`: L'image d'entrée sur laquelle les masques doivent être superposés.
"""
function show_anns_jl(anns, image)
    if length(anns) == 0
        return
    end

    sorted_anns = sort(anns, by=x -> x["area"], rev=true)

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_rgb = float.(image) / 255.0

    alpha = 0.5 # transparency (0 = transparent, 1 = opaque) 

    img_with_masks = copy(img_rgb) # copy image to apply masks

    for ann in sorted_anns
        if haskey(ann, "segmentation")
            m = ann["segmentation"]

            if typeof(m) <: AbstractArray
                mask_color = rand(RGB)
                mask = m .> 0  # Boolean conversion fo the mask

                img_with_masks[:, :, 1] .= img_with_masks[:, :, 1] .* (1 .- alpha .* mask) .+ alpha * mask_color.r .* mask  # red
                img_with_masks[:, :, 2] .= img_with_masks[:, :, 2] .* (1 .- alpha .* mask) .+ alpha * mask_color.g .* mask  # green
                img_with_masks[:, :, 3] .= img_with_masks[:, :, 3] .* (1 .- alpha .* mask) .+ alpha * mask_color.b .* mask  # blue
            else
                println("Error: bad type for 'segmentation'.")
            end
        else
            println("Error: 'segmentation' key not found.")
        end
    end

    # convert image with RGB{Float64}
    img_final_rgb = colorview(RGB, img_with_masks[:, :, 1], img_with_masks[:, :, 2], img_with_masks[:, :, 3])

    display(plot(img_final_rgb, axis=false, legend=false))
end

end # module SegmentAnything