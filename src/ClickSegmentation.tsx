import {
  executeOperator,
  Operator,
  OperatorConfig,
  registerOperator,
} from "@fiftyone/operators";
import * as fos from "@fiftyone/state";
import _ from "lodash";
import { useRecoilValue } from "recoil";
import styled from "styled-components";
import { useState, useRef, useCallback } from "react";

export function ClickSegmentation() {
  const modalSample = useRecoilValue(fos.modalSample);
  const [clicks, setClicks] = useState([]);
  const [fieldName, setFieldName] = useState("user_clicks");
  const [modelName, setModelName] = useState("segment-anything-2-hiera-small-image-torch");
  const [labelName, setLabelName] = useState("label");
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  const handleImageClick = (event) => {
    if (!imageRef.current || !containerRef.current) return;
    
    const img = imageRef.current;
    const container = containerRef.current;
    const imgRect = img.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    
    const x = event.clientX - containerRect.left;
    const y = event.clientY - containerRect.top;
    const imgX = event.clientX - imgRect.left;
    const imgY = event.clientY - imgRect.top;
    
    const displayWidth = imgRect.width;
    const displayHeight = imgRect.height;
    const normalizedX = imgX / displayWidth;
    const normalizedY = imgY / displayHeight;
  
    const clickData = {
      displayX: x,
      displayY: y,
      normalizedX: parseFloat(normalizedX.toFixed(4)),
      normalizedY: parseFloat(normalizedY.toFixed(4)),
    };
    
    console.log("Click captured:", clickData);
    setClicks((prev) => [...prev, clickData]);
  };
  
  const saveAsKeypoints = async () => {
    if (clicks.length === 0) {
      alert("No clicks to save");
      return;
    }
    
    if (!modalSample) {
      alert("No sample selected");
      return;
    }
    
    if (!fieldName.trim()) {
      alert("Please enter a field name");
      return;
    }

    const sampleId = modalSample.sample._id;
    const keypointCoords = clicks.map(click => [
      click.normalizedX,
      click.normalizedY,
    ]);
    
    try {
      await executeOperator(
        "@51labs/click-segmentation/save_keypoints",
        {
          sample_id: sampleId,
          keypoints: keypointCoords,
          field_name: fieldName.trim(),
          label_name: labelName.trim()
        }
      );
      setClicks([]);
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    }
  };

  const segmentWithKeypoints = async () => {
    if (!modalSample) {
      alert("No sample selected");
      return;
    }
    
    if (!modelName.trim()) {
      alert("Please enter a model name");
      return;
    }

    const sampleId = modalSample.sample._id;    
    try {
      alert("Segmentation started ...");
      await executeOperator(
        "@51labs/click-segmentation/segment_with_keypoints",
        {
          sample_id: sampleId,
          keypoints_field: fieldName.trim(),
          model_name: modelName.trim()
        }
      );
    } catch (error) {
      console.error("Error saving keypoints:", error);
      alert(`Failed: ${error.message}`);
    }
  };

  const clearClicks = () => {
    setClicks([]);
  };
  
  const sample = modalSample.sample;
  const filepath = sample?.filepath;
  const mediaUrl = `/media?filepath=${encodeURIComponent(filepath)}`;
  
  return (
    <div style={{ padding: "20px", display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ marginBottom: "10px" }}>
        <h3 style={{ margin: 0 }}>Image segmentation via point prompts</h3>
        <p style={{ fontSize: "12px", color: "#666", margin: "5px 0" }}>
          Click on the image to capture coordinates
        </p>
      </div>
      
      {/* Image Viewer with Click Markers */}
      <div 
        ref={containerRef}
        style={{ 
          width: "auto", 
          height: "800px",
          position: "relative",
          backgroundColor: "#1e1e1e",
          borderRadius: "4px",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "20px"
        }}
        onClick={handleImageClick}
      >
        <img 
          ref={imageRef}
          src={mediaUrl}
          alt="Sample"
          style={{ 
            maxWidth: "200%",
            maxHeight: "auto",
            objectFit: "contain",
            cursor: "crosshair",
            userSelect: "none"
          }}
          onLoad={() => console.log("Image loaded successfully")}
          onError={(e) => console.error("Image failed to load", e)}
        />
        
        {/* Render clicks as dots on the image */}
        {clicks.map((click, index) => (
          <div
            key={click.id}
            style={{
              position: "absolute",
              left: `${click.displayX}px`,
              top: `${click.displayY}px`,
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: "#ff4444",
              border: "2px solid white",
              transform: "translate(-50%, -50%)",
              pointerEvents: "none",
              boxShadow: "0 0 4px rgba(0,0,0,0.5)",
              zIndex: 10
            }}
          >
          </div>
        ))}
      </div>
      
      {/* Field Name Input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Sample field name for saving clicks (as keypoints)
          </span>
          <input 
            type="text"
            value={fieldName}
            onChange={(e) => setFieldName(e.target.value)}
            placeholder="e.g., user_clicks, keypoints"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      {/* Keypoint Label Name Input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Label name for the current set of clicks
          </span>
          <input 
            type="text"
            value={labelName}
            onChange={(e) => setLabelName(e.target.value)}
            placeholder="e.g., animal, person"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      {/* Keypoint Buttons */}
      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
      <button 
          onClick={saveAsKeypoints}
          disabled={clicks.length === 0}
          style={{
            padding: "8px 16px",
            backgroundColor: clicks.length > 0 ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: clicks.length > 0 ? "pointer" : "not-allowed",
            fontWeight: "bold"
          }}
        >
          Save as Keypoints ({clicks.length})
        </button>
        <button 
          onClick={clearClicks}
          style={{
            padding: "8px 16px",
            backgroundColor: "#ff5555",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Clear Clicks ({clicks.length})
        </button>
      </div>

      {/* Model Name Input */}
      <div style={{ 
        marginBottom: "8px", 
        padding: "4px",
      }}>
        <label style={{ 
          display: "flex", 
          flexDirection: "column", 
          gap: "8px",
          fontSize: "14px"
        }}>
          <span style={{ fontSize: "14px", color: "#666" }}>
            Promptable segmentation model from FiftyOne model zoo
          </span>
          <input 
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., segment-anything-2-hiera-small-image-torch"
            style={{
              padding: "8px 12px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "14px",
              fontFamily: "monospace"
            }}
          />
        </label>
      </div>

      <div style={{ marginBottom: "10px", display: "flex", gap: "10px" }}>
      <button 
          onClick={segmentWithKeypoints}
          disabled={clicks.length > 0}
          style={{
            padding: "8px 16px",
            backgroundColor: clicks.length > 0 ? "#ccc" : "#2196F3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: clicks.length == 0 ? "pointer" : "not-allowed",
            fontWeight: "bold"
          }}
        >
          Segment with keypoints
        </button>
      </div>
    </div>

    
  );
}