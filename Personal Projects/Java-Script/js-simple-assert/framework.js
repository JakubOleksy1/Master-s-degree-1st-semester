var assert = function(a,b){
    if(a === b){
        console.log("Test passed!", a, "===", b);
    } else {
        console.error("Test failed!", a, "!==", b);
    }
};