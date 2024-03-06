from mmocr.apis import MMOCRInferencer
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
ocr('demo/demo_kie.jpeg', show=True, print_result=True)

#ocr = MMOCRInferencer(det='DBNetpp', rec='ABINet_Vision')
#ocr('demo/demo_kie.jpeg', show=True,draw_pred=True,print_result=True)