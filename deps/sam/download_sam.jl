using Downloads


model_dir = @__DIR__

if !isdir(model_dir)
    println("Création du dossier pour stocker les modèles SAM dans deps/sam...")
    mkpath(model_dir)
end


models = Dict(
    "sam_vit_l_0b3195.pth" => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_h_4b8939.pth" => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_b_01ec64.pth" => "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)


function download_if_absent(model_name, model_url)
    model_path = joinpath(model_dir, model_name)
    
    if !isfile(model_path)
        println("Modèle $model_name absent. Téléchargement en cours...")
        Downloads.download(model_url, model_path)
        println("Modèle $model_name téléchargé et sauvegardé dans $model_path.")
    else
        println("Modèle $model_name déjà présent dans $model_path.")
    end
end


for (model_name, model_url) in models
    download_if_absent(model_name, model_url)
end

println("Tous les modèles SAM sont disponibles dans $model_dir.")