using Plots, LaTeXStrings, ColorSchemes
using Random

include("OFDMSerial.jl")
include("OFDMParallel.jl")

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
gr()

function run_serial(msglen)
    @info "Message length = $msglen SERIAL"
    stream = bitrand(msglen)
    
    blocks = OFDMSerial.to_blocks(stream)
    
    @info "Interleave"
    t_interleave = @elapsed begin
        i_blocks = OFDMSerial.interleave.(blocks)
    end
    @info "Elapsed time: $t_interleave"

    @info "Modulation"        
    t_modulation = @elapsed begin
        m_blocks = OFDMSerial.modulate_block.(i_blocks)
    end
    @info "Elapsed time: $t_modulation"   

    @info "IFFT"
    t_ifft = @elapsed begin
        waves = OFDMSerial.to_waveform.(m_blocks)
    end
    @info "Elapsed time: $t_interleave"
    
    return t_interleave, t_modulation, t_ifft
end

function run_parallel(msglen)
    @info "Message length = $msglen PARALLEL"
    stream = OFDMParallel.zero_pad(bitrand(msglen))
    
    @info "Interleave"
    t_interleave = @elapsed begin
        i_stream = OFDMParallel.interleave(stream)
    end
    @info "Elapsed time: $t_interleave"

    @info "Modulation"    
    t_modulation = @elapsed begin
        m_stream = OFDMParallel.modulate_block(i_stream)
    end
    @info "Elapsed time: $t_modulation"   

    @info "IFFT"
    t_ifft = @elapsed begin
        waves = OFDMParallel.to_waveform(m_stream)
    end
    @info "Elapsed time: $t_interleave"

    return t_interleave, t_modulation, t_ifft
end

function plot_timings(bitlens)
    t_interleave_s = Float64[]
    t_modulation_s = Float64[]
    t_ifft_s = Float64[]
    t_interleave_p = Float64[]
    t_modulation_p = Float64[]
    t_ifft_p = Float64[]
    
    for bitlen in bitlens
        interleave_time_s, modulation_time_s, ifft_time_s = run_serial(bitlen)
        interleave_time_p, modulation_time_p, ifft_time_p = run_parallel(bitlen)
        push!(t_interleave_s, interleave_time_s)
        push!(t_modulation_s, modulation_time_s)
        push!(t_ifft_s, ifft_time_s)
        push!(t_interleave_p, interleave_time_p)
        push!(t_modulation_p, modulation_time_p)
        push!(t_ifft_p, ifft_time_p)
    end

    t_interleave_s = t_interleave_s[2:end]
    t_modulation_s = t_modulation_s[2:end]
    t_ifft_s = t_ifft_s[2:end]
    t_interleave_p = t_interleave_p[2:end]
    t_modulation_p = t_modulation_p[2:end]
    t_ifft_p = t_ifft_p[2:end]
    bitlens = bitlens[2:end]
    
    p1 = plot(xlabel="N. Bits", ylabel="Tiempo (segundos)", title="Tiempos Serie", legend=:topleft)
    plot!(bitlens, t_interleave_s, label="Entrelazado")
    plot!(bitlens, t_modulation_s, label="Modulación")
    plot!(bitlens, t_ifft_s, label="IFFT")
    savefig("times_s.svg")
    p2 = plot(xlabel="N. Bits", ylabel="Tiempo (segundos)", title="Tiempos Paralelo", legend=:topleft)
    plot!(bitlens, t_interleave_p, label="Entrelazado")
    plot!(bitlens, t_modulation_p, label="Modulación")
    plot!(bitlens, t_ifft_p, label="IFFT")
    savefig("times_p.svg")
    p3 = plot(xlabel="N. Bits", ylabel="Tiempo (segundos)", title="Tiempos Entrelazado", legend=:topleft)
    plot!(bitlens, t_interleave_p, label="Paralelo")
    plot!(bitlens, t_interleave_s, label="Serie")
    savefig("times_inter.svg")
    p4 = plot(xlabel="N. Bits", ylabel="Tiempo (segundos)", title="Tiempos Modulación", legend=:topleft)
    plot!(bitlens, t_modulation_p, label="Paralelo")
    plot!(bitlens, t_modulation_s, label="Serie")
    savefig("times_mod.svg")
    p5 = plot(xlabel="N. Bits", ylabel="Tiempo (segundos)", title="Tiempos IFFT", legend=:topleft)
    plot!(bitlens, t_ifft_p, label="Paralelo")
    plot!(bitlens, t_ifft_s, label="Serie")
    savefig("times_ifft.svg")
    return
end

bitlens = 10 .^ (1:8) .|> Int

plot_timings(bitlens)
