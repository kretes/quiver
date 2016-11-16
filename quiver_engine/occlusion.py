from imagenet_utils import decode_predictions
import numpy as np

def occlusion_positions(size, img_size, overlap=0.0):
    olf = 1 / (1 - overlap)
    imgf = int(np.ceil(img_size / size))
    return [int(i * size * (1 - overlap)) for i in range(int(olf * imgf)) if int(i * size * (1 - overlap)) < img_size]

def generate_img_with_occlusion(img,x1,x2,y1,y2,occlusion_val=0):
    imgc = np.copy(img)
    imgc[y1:y2,x1:x2,:] = occlusion_val
    return imgc

def occlude_patches_and_predict(model, img,occ_size_w,occ_size_h,overlap=0.0):
    for x in occlusion_positions(occ_size_h,img.shape[0],overlap):
        for y in occlusion_positions(occ_size_w,img.shape[1],overlap):
            imgc = generate_img_with_occlusion(img,x,x+occ_size_w,y,y+occ_size_h)
            img_for_model = np.expand_dims(imgc,0)
            pred = model.predict(img_for_model, batch_size=1, verbose=0)
            decoded = decode_predictions(pred)[0]
            yield({'pred':decoded,'x':x,'y':y})

def occlude_and_predict(model,image_for_model):
    img = image_for_model[0]
    occ_size_w = int(np.floor(img.shape[0]/3))
    occ_size_h = int(np.floor(img.shape[1]/3))
    occluded_predictions = list(occlude_patches_and_predict(model,img,occ_size_w,occ_size_h,0.5))
    baseline_pred = decode_predictions(model.predict(image_for_model))[0]
    prediction_diffs = [predictions_diff_top(baseline_pred, occluded_prediction['pred']) for occluded_prediction in occluded_predictions]
    results_diffs = [{'x': r['x'], 'y': r['y'], 'diff': diff} for r, diff in zip(occluded_predictions, prediction_diffs)]
    return generate_mask(results_diffs,img.shape[0], img.shape[1],occ_size_w,occ_size_h)

def generate_mask(x_y_and_diff,width,height,occ_size_w,occ_size_h):
    mask = np.zeros((width,height, 3))
    for occlusion_result in x_y_and_diff:
        y = occlusion_result['y']
        x = occlusion_result['x']
        if (occlusion_result['diff'] > 0):
            mask[y:y + occ_size_h, x:x + occ_size_w, :] += occlusion_result['diff']
    return mask * (1.0 / mask.max())

def predictions_diff_top(pred_1,pred_2):
    pred_1 = np.array(pred_1)
    pred_2 = np.array(pred_2)
    top_code_from_1 = pred_1[0][0]
    top_val_from_1 = pred_1[0][2]
    predictions_for_top_in_2 = pred_2[pred_2[:,0] == top_code_from_1]
    if (len(predictions_for_top_in_2) == 0):
        top_val_in_2 = 0
    else:
        top_val_in_2 = predictions_for_top_in_2[0][2]
    return float(top_val_from_1)-float(top_val_in_2)

