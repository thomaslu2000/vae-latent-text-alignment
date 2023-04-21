from image_similarity_measures.evaluate import evaluation
from PIL import Image, ImageOps

def metricx(predicted_image_path,gt_image_path):
    # predicted_image_path = "C:/Users/DELL/Desktop/Project/Project/Output/Output_1.png"
    # gt_image_path = "C:/Users/DELL/Desktop/Project/Project/Output/Output_1.png"


  MM = ["issm","psnr","rmse","sam","sre","ssim"]
  res = []
    
  for m in MM:    
    e = evaluation(org_img_path=gt_image_path, pred_img_path=predicted_image_path, metrics=[m])
    res.append(e[m])
    

  image = Image.open(predicted_image_path)
  image = ImageOps.grayscale(image)
  image.save(predicted_image_path)


  image = Image.open(gt_image_path)
  image = ImageOps.grayscale(image)
  image.save(gt_image_path)


  for m in MM:
    e = evaluation(org_img_path=gt_image_path,  pred_img_path=predicted_image_path,  metrics=[m])
    res.append(e[m])

  return res
