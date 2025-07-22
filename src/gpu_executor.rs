// src/gpu_executor.rs
use wgpu::util::DeviceExt;
use anyhow::{Result, Context};
use crate::{GNBGProblem, GNBGGpuData, shaders::GNBG_COMPUTE_SHADER};

pub struct GpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    gpu_data: GNBGGpuData,
    dimension: u32,
}

impl GpuExecutor {
    pub async fn new(problem: &GNBGProblem) -> Result<Self> {
        // Initialize GPU
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("Failed to find GPU adapter")?;
            
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GNBG Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
            
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GNBG Shader"),
            source: wgpu::ShaderSource::Wgsl(GNBG_COMPUTE_SHADER.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GNBG Bind Group Layout"),
            entries: &[
                // Solutions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Component params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Component data
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mu/Omega data
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Fitness output
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GNBG Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GNBG Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        let gpu_data = problem.to_gpu_format()?;
        let dimension = problem.dimension as u32;
        
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            gpu_data,
            dimension,
        })
    }
    
    pub async fn evaluate_batch(&self, solutions: &[f32]) -> Result<Vec<f32>> {
        let n_solutions = solutions.len() as u32 / self.dimension;
        
        if solutions.len() != (n_solutions * self.dimension) as usize {
            anyhow::bail!("Solution buffer size mismatch");
        }
        
        // Create buffers
        let solution_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solution Buffer"),
            contents: bytemuck::cast_slice(solutions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&self.gpu_data.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let components_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Components Buffer"),
            contents: bytemuck::cast_slice(&self.gpu_data.component_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let component_data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Component Data Buffer"),
            contents: bytemuck::cast_slice(&self.gpu_data.flattened_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let mu_omega_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mu Omega Buffer"),
            contents: bytemuck::cast_slice(&self.gpu_data.mu_omega_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let fitness_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Buffer"),
            size: (n_solutions * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n_solutions * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GNBG Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: solution_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: components_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: component_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mu_omega_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: fitness_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GNBG Encoder"),
        });
        
        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GNBG Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (n_solutions + 63) / 64; // 64 = workgroup size
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(
            &fitness_buffer,
            0,
            &staging_buffer,
            0,
            (n_solutions * 4) as u64,
        );
        
        // Submit commands
        self.queue.submit(Some(encoder.finish()));
        
        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        rx.await??;
        
        let data = buffer_slice.get_mapped_range();
        let fitness: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(fitness)
    }
}
