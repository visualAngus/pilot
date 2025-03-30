class DepthMapGenerator:
    def __init__(self, model_type="DPT_Hybrid"):
        self.model_type = model_type
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

    def compute_depth_map(self, image_path, output_path="depth_map.png"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)

        cv2.imwrite(output_path, depth_map)
        return depth_map

    def enhance_contrast(self, depth_map, lower_bound, upper_bound):
        depth_map = cv2.normalize(depth_map, None, alpha=lower_bound, beta=upper_bound, norm_type=cv2.NORM_MINMAX)
        return depth_map

    def select_distance_range(self, depth_map, lower_bound, upper_bound):
        mask = (depth_map >= lower_bound) & (depth_map <= upper_bound)
        filtered_depth_map = np.zeros_like(depth_map)
        filtered_depth_map[mask] = depth_map[mask]
        return filtered_depth_map