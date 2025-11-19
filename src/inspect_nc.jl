using NCDatasets
ds = NCDataset("verification_bcs.nc")
println("Variables: ", keys(ds))
println("T dims: ", dimnames(ds["T"]))
println("v dims: ", dimnames(ds["v"]))
println("T size: ", size(ds["T"]))
println("v size: ", size(ds["v"]))
println("y_aca size: ", size(ds["y_aca"]))
println("y_afa size: ", size(ds["y_afa"]))
close(ds)



