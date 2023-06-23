using Plots, LaTeXStrings, ColorSchemes
include("OFDMSerial.jl")

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
pyplot()

function run_main(msglen)
    stream = bitrand(msglen)
    
    blocks = OFDMSerial.to_blocks(stream)
    
    t_interleave = @elapsed begin
        blocks = OFDMSerial.interleave.(blocks, scheme)
    end
    
    t_modulation = @elapsed begin
        blocks = OFDMSerial.modulate_block.(blocks, scheme)
    end
    
    t_ifft = @elapsed begin
        waves = OFDMSerial.to_waveform.(sblocks)
    end
    
    return t_interleave, t_modulation, t_ifft
end

function plot_timings(bitlens, scheme)
    t_interleave = Float64[]
    t_modulation = Float64[]
    t_ifft = Float64[]
    
    for bitlen in bitlens
        interleave_time, modulation_time, ifft_time = run_main(bitlen, scheme)
        push!(t_interleave, interleave_time)
        push!(t_modulation, modulation_time)
        push!(t_ifft, ifft_time)
    end
    
    p = plot(xlabel="Bit Length", ylabel="Time (seconds)", legend=:topleft)
    plot!(bitlens, t_interleave, label="Interleave")
    plot!(bitlens, t_modulation, label="Modulation")
    plot!(bitlens, t_ifft, label="IFFT")
    return p
end

bitlens = 10 .^ (2:8) .|> Int  # Example bit lengths

p1 = plot_timings(bitlens, :qam16)

savefig("times.svg")
