from image_similarity_measures.evaluate import evaluation
from PIL import Image, ImageOps
import cv2

def metricx(predicted_image_path,gt_image_path):
    # predicted_image_path = "C:/Users/DELL/Desktop/Project/Project/Output/Output_1.png"
    # gt_image_path = "C:/Users/DELL/Desktop/Project/Project/Output/Output_1.png"


  MM = ["issm","psnr","rmse","sam","sre","ssim"]
  res = []

  image_A = cv2.imread(gt_image_path).astype("float32")

  image_B = cv2.imread(predicted_image_path).astype("float32")

  image_A = cv2.cvtColor(image_A,cv2.COLOR_BGR2LAB) 
  image_B = cv2.cvtColor(image_B,cv2.COLOR_BGR2LAB) 

  diff = cv2.add(image_A,-image_B)

  diff_L = diff[:,:,0]
  diff_A = diff[:,:,1]
  diff_B = diff[:,:,2]

  delta_e = np.mean( np.sqrt(diff_L*diff_L + diff_A*diff_A + diff_B*diff_B) )

  res.append(delta_e)
    
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
