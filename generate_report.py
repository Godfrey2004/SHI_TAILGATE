import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_inspection_report(serial_number, status, date_str, time_str, 
                             zone_images, ocr_serial, confidence, 
                             defects, output_path):
    """
    Generate a PDF report for the tailgate inspection.
    zone_images: dict mapping zone name to image path (e.g. {'inner1': 'path/to/img.jpg', ...})
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 24)
    if status == "PASS":
        c.setFillColor(colors.darkgreen)
    else:
        c.setFillColor(colors.red)
    c.drawString(50, height - 50, f"Tailgate Inspection Report - {status}")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Date: {date_str} {time_str}")
    c.drawString(50, height - 100, f"Serial Number: {serial_number}")
    c.drawString(50, height - 120, f"OCR Confidence: {confidence}")
    
    if defects:
        c.setFillColor(colors.red)
        c.drawString(50, height - 140, f"Defects: {', '.join(defects)}")
    else:
        c.setFillColor(colors.darkgreen)
        c.drawString(50, height - 140, "Defects: None")
        
    c.setFillColor(colors.black)
    
    # Draw images
    y_position = height - 180
    image_width = 220
    image_height = 160
    
    positions = {
        "ocr_crop": ((width - image_width) / 2, y_position - image_height),
        "inner1": (50, y_position - 2 * image_height - 40),
        "inner2": (300, y_position - 2 * image_height - 40),
        "outer1": (50, y_position - 3 * image_height - 80),
        "outer2": (300, y_position - 3 * image_height - 80)
    }
    
    zones = ["ocr_crop", "inner1", "inner2", "outer1", "outer2"]
    
    for zone in zones:
        img_path = zone_images.get(zone)
        x, y = positions.get(zone, (0, 0))
        
        c.setFont("Helvetica-Bold", 12)
        label_text = "OCR SERIAL CROP" if zone == "ocr_crop" else f"Zone: {zone.upper()}"
        c.drawString(x, y + image_height + 10, label_text)
        
        if img_path and os.path.exists(img_path):
            try:
                c.drawImage(img_path, x, y, width=image_width, height=image_height, preserveAspectRatio=True)
            except Exception as e:
                c.setStrokeColor(colors.red)
                c.rect(x, y, image_width, image_height)
                c.drawString(x + 10, y + image_height/2, "Error loading image")
        else:
            c.setStrokeColor(colors.gray)
            c.rect(x, y, image_width, image_height)
            c.drawString(x + 10, y + image_height/2, "No Image Available")

    c.save()
    print(f"Report generated successfully: {output_path}")
