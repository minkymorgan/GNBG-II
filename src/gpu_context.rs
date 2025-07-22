/*
 * GNBG-II GPU-Accelerated Implementation - Shared GPU Context
 * Copyright (C) 2025 Andrew Morgan <minkymorgan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

use wgpu::util::DeviceExt;
use anyhow::{Result, Context};
use std::sync::Arc;

/// Shared GPU context for GNBG single-objective and multi-objective problems
/// 
/// This provides a centralized GPU resource management system that can be
/// shared across different problem types, avoiding duplicate GPU initialization
/// and enabling efficient resource utilization.
#[derive(Clone)]
pub struct GpuContext {
    /// GPU device handle
    pub device: Arc<wgpu::Device>,
    /// GPU command queue
    pub queue: Arc<wgpu::Queue>,
    /// GPU adapter information
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new shared GPU context
    /// 
    /// Initializes the GPU device and queue that can be shared across
    /// different executors and problem types.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find GPU adapter")?;
            
        let adapter_info = adapter.get_info();
        log::info!("Using GPU: {} ({:?})", adapter_info.name, adapter_info.backend);
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GNBG Shared GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
            
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
    
    /// Create a compute pipeline with the given shader and bind group layout
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        })
    }
    
    /// Create a buffer with initial data
    pub fn create_buffer_init(&self, descriptor: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        self.device.create_buffer_init(descriptor)
    }
    
    /// Create an empty buffer
    pub fn create_buffer(&self, descriptor: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(descriptor)
    }
    
    /// Create a bind group layout
    pub fn create_bind_group_layout(
        &self, 
        descriptor: &wgpu::BindGroupLayoutDescriptor
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(descriptor)
    }
    
    /// Create a bind group
    pub fn create_bind_group(&self, descriptor: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(descriptor)
    }
    
    /// Create a command encoder
    pub fn create_command_encoder(
        &self,
        descriptor: &wgpu::CommandEncoderDescriptor,
    ) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(descriptor)
    }
    
    /// Submit commands to the GPU queue
    pub fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(&self, command_buffers: I) {
        self.queue.submit(command_buffers);
    }
    
    /// Poll the device for completion
    pub fn poll(&self, maintain: wgpu::Maintain) {
        self.device.poll(maintain);
    }
    
    /// Get device limits
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }
    
    /// Get adapter information
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
}

/// GPU context manager for sharing GPU resources
/// 
/// Provides singleton-like access to GPU context while allowing
/// multiple references across different parts of the system.
pub struct GpuContextManager {
    context: Option<GpuContext>,
}

impl GpuContextManager {
    /// Create a new context manager
    pub fn new() -> Self {
        Self { context: None }
    }
    
    /// Get or create the shared GPU context
    pub async fn get_context(&mut self) -> Result<GpuContext> {
        if self.context.is_none() {
            self.context = Some(GpuContext::new().await?);
        }
        Ok(self.context.as_ref().unwrap().clone())
    }
    
    /// Check if context is initialized
    pub fn is_initialized(&self) -> bool {
        self.context.is_some()
    }
}

impl Default for GpuContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_context_creation() {
        let context = GpuContext::new().await;
        
        // On systems without GPU, this might fail, so we'll make it optional
        if context.is_ok() {
            let ctx = context.unwrap();
            println!("GPU: {}", ctx.adapter_info().name);
            assert!(!ctx.adapter_info().name.is_empty());
        }
    }
    
    #[tokio::test]
    async fn test_context_manager() {
        let mut manager = GpuContextManager::new();
        assert!(!manager.is_initialized());
        
        // This might fail on systems without GPU
        if let Ok(context1) = manager.get_context().await {
            assert!(manager.is_initialized());
            
            let context2 = manager.get_context().await.unwrap();
            
            // Should be the same Arc references
            assert!(Arc::ptr_eq(&context1.device, &context2.device));
            assert!(Arc::ptr_eq(&context1.queue, &context2.queue));
        }
    }
}