package com.vectordb.util;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

public class FileParser {

    public static String parseText(String text) {
        return text == null ? "" : text.trim();
    }

    public static String parsePdf(byte[] pdfBytes) {
        try (PDDocument doc = Loader.loadPDF(pdfBytes)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(doc).trim();
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract PDF text: " + e.getMessage(), e);
        }
    }
}
