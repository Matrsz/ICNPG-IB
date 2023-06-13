using Plots, LaTeXStrings, ColorSchemes
using FFTW
include("OFDM_serial.jl")

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
pyplot()

function run_main(msglen, scheme)
    stream = bitrand(msglen)
    
    t_to_blocks = @elapsed begin
        blocks = to_blocks(stream, scheme)
    end
    
    t_interleave = @elapsed begin
        blocks = interleave.(blocks, scheme)
    end
    
    t_modulation = @elapsed begin
        blocks = modulate_block.(blocks, scheme)
    end
    
    t_pilot_waves = @elapsed begin
        pblocks = add_pilots.(blocks)
        sblocks = shift.(pblocks)
    end
    
    t_ifft = @elapsed begin
        waves = ifft.(sblocks)
    end
    
    return t_to_blocks, t_interleave, t_modulation, t_pilot_waves, t_ifft
end

function plot_timings(bitlens, scheme)
    t_interleave = Float64[]
    t_modulation = Float64[]
    t_pilot_waves = Float64[]
    t_ifft = Float64[]
    
    for bitlen in bitlens
        to_blocks_time, interleave_time, modulation_time, pilot_waves_time, ifft_time = run_main(bitlen, scheme)
        push!(t_interleave, interleave_time)
        push!(t_modulation, modulation_time)
        push!(t_pilot_waves, pilot_waves_time)
        push!(t_ifft, ifft_time)
    end
    
    p = plot(xlabel="Bit Length", ylabel="Time (seconds)", legend=:topleft)
    plot!(bitlens, t_interleave, label="Interleave")
    plot!(bitlens, t_modulation, label="Modulation")
    plot!(bitlens, t_pilot_waves, label="Pilot Waves")
    plot!(bitlens, t_ifft, label="IFFT")
    return p
end

bitlens = 10 .^ (2:8) .|> Int  # Example bit lengths

p1 = plot_timings(bitlens, :qam16)

savefig("times.svg")
