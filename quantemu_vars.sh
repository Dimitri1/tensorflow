# Convolution 
ENABLE_QUANTOP_CONV=1
ENABLE_QUANTOP_CONV_GRAD=0
ENABLE_QUANTOP_CONV_WTGRAD=0

# BatchNormalization
ENABLE_QUANTOP_BNORM=1
ENABLE_QUANTOP_BNORM_NORM_ONLY=0
ENABLE_QUANTOP_BNORM_GRAD=0

# DENSE  
ENABLE_QUANTOP_DENSE=1
ENABLE_QUANTOP_DENSE_GRAD=0

# MATMUL 
ENABLE_QUANTOP_MATMUL=1
ENABLE_QUANTOP_MATMUL_GRAD=0

# MUL OP  
ENABLE_QUANTOP_MUL=1
ENABLE_QUANTOP_MUL_GRAD=0

# NON-LINEAR OPs
ENABLE_QUANTOP_SIGMOID=0
ENABLE_QUANTOP_SIGMOID_GRAD=0
ENABLE_QUANTOP_TANH=0
ENABLE_QUANTOP_TANH_GRAD=0

# Data Type Settings 
# INT=1, UINT=2, LOWP_FP=3, LOG2=4, POSIT=5, BFLOAT(RNE,STOCHASTIC)=6, MODFP16=7  
QUANTEMU_INPUT_DATA_TYPE=3
QUANTEMU_FILTER_DATA_TYPE=3
QUANTEMU_GRAD_DATA_TYPE=3
QUANTEMU_WTGRAD_DATA_TYPE=3

QUANTEMU_DENSE_DATA_TYPE=3 
QUANTEMU_BNORM_DATA_TYPE=3
QUANTEMU_MUL_DATA_TYPE=3
QUANTEMU_TANH_DATA_TYPE=3
QUANTEMU_SIGMOID_DATA_TYPE=3

# only used by LOWP_FP, POSIT and BLOCK_FP types 
QUANTEMU_EXPBITS=5

# Rounding modes  
# Truncate (no rounding)=0, Round to Nearest Even(RNE)=1, STOCHASTIC_ROUNDING=2  
QUANTEMU_RMODE_INPUTS=2
QUANTEMU_RMODE_FILTERS=1
QUANTEMU_RMODE_GRADS=2
QUANTEMU_RMODE_WTGRADS=1
QUANTEMU_BNORM_RMODE_INPUTS=1
QUANTEMU_BNORM_RMODE_GRADS=1

# Precision Settings 
QUANTEMU_FIRST_LAYER_PRECISION=16

QUANTEMU_PRECISION_CONV_INPUTS=8
QUANTEMU_PRECISION_CONV_FILTERS=8
QUANTEMU_PRECISION_CONV_GRADS=8
QUANTEMU_PRECISION_CONV_WTGRADS=8

QUANTEMU_PRECISION_DENSE_INPUTS=8
QUANTEMU_PRECISION_DENSE_FILTERS=8
QUANTEMU_PRECISION_DENSE_GRADS=8

QUANTEMU_PRECISION_BNORM_INPUTS=16
QUANTEMU_PRECISION_BNORM_GRADS=16
QUANTEMU_PRECISION_MATMUL_INPUTS=8
QUANTEMU_PRECISION_MATMUL_FILTERS=8
QUANTEMU_PRECISION_MATMUL_GRADS=8

QUANTEMU_PRECISION_MUL_INPUTS=8
QUANTEMU_PRECISION_MUL_GRADS=8
QUANTEMU_PRECISION_SIGMOID_INPUTS=3 
QUANTEMU_PRECISION_SIGMOID_GRADS=3 
QUANTEMU_PRECISION_TANH_INPUTS=3 
QUANTEMU_PRECISION_TANH_GRADS=3 

# Buffer Copy Settings 
# Make a Copy while Quantizing 
QUANTEMU_ALLOCATE_COPY_INPUTS=0
QUANTEMU_ALLOCATE_COPY_FILTERS=0
QUANTEMU_ALLOCATE_COPY_GRADS=0

# FGQ Settings 
# NOBLOCK=0, BLOCK_C=1, BLOCK_CHW=2 
QUANTEMU_CBLOCK_TYPE_CONV_INPUTS=0
QUANTEMU_CBLOCK_TYPE_CONV_FILTERS=0
QUANTEMU_CBLOCK_TYPE_CONV_GRADS=0
QUANTEMU_CBLOCK_TYPE_CONV_WTGRADS=0

QUANTEMU_CBLOCK_TYPE_BNORM_INPUTS=0
QUANTEMU_CBLOCK_TYPE_BNORM_GRADS=0

QUANTEMU_CBLOCK_SIZE_INPUTS=2048
QUANTEMU_CBLOCK_SIZE_FILTER=2048
QUANTEMU_CBLOCK_SIZE_GRAD=2048
QUANTEMU_CBLOCK_SIZE_WTGRAD=2048

export ENABLE_QUANTOP_CONV
export ENABLE_QUANTOP_CONV_GRAD
export ENABLE_QUANTOP_CONV_WTGRAD
export QUANTEMU_WTGRAD_DATA_TYPE

export QUANTEMU_MUL_DATA_TYPE
export QUANTEMU_BNORM_DATA_TYPE
export QUANTEMU_DENSE_DATA_TYPE
export QUANTEMU_TANH_DATA_TYPE
export QUANTEMU_SIGMOID_DATA_TYPE

export QUANTEMU_RMODE_WTGRADS
export QUANTEMU_PRECISION_CONV_WTGRADS
export QUANTEMU_CBLOCK_TYPE_CONV_WTGRADS
export QUANTEMU_CBLOCK_SIZE_WTGRAD

export ENABLE_QUANTOP_BNORM
export ENABLE_QUANTOP_BNORM_NORM_ONLY
export ENABLE_QUANTOP_BNORM_GRAD
export ENABLE_QUANTOP_DENSE
export ENABLE_QUANTOP_DENSE_GRAD

export QUANTEMU_BNORM_RMODE_INPUTS
export QUANTEMU_BNORM_RMODE_GRADS

export ENABLE_QUANTOP_MATMUL
export ENABLE_QUANTOP_MATMUL_GRAD
export ENABLE_QUANTOP_MUL
export ENABLE_QUANTOP_MUL_GRAD

export ENABLE_QUANTOP_SIGMOID
export ENABLE_QUANTOP_SIGMOID_GRAD
export ENABLE_QUANTOP_TANH
export ENABLE_QUANTOP_TANH_GRAD

export QUANTEMU_INPUT_DATA_TYPE
export QUANTEMU_FILTER_DATA_TYPE
export QUANTEMU_GRAD_DATA_TYPE
export QUANTEMU_EXPBITS          
export QUANTEMU_RMODE_INPUTS
export QUANTEMU_RMODE_FILTERS
export QUANTEMU_RMODE_GRADS
export QUANTEMU_FIRST_LAYER_PRECISION
export QUANTEMU_PRECISION_CONV_INPUTS
export QUANTEMU_PRECISION_CONV_FILTERS
export QUANTEMU_PRECISION_CONV_GRADS
export QUANTEMU_PRECISION_DENSE_INPUTS
export QUANTEMU_PRECISION_DENSE_FILTERS
export QUANTEMU_PRECISION_DENSE_GRADS
export QUANTEMU_PRECISION_BNORM_INPUTS
export QUANTEMU_PRECISION_BNORM_GRADS
export QUANTEMU_PRECISION_MATMUL_INPUTS
export QUANTEMU_PRECISION_MATMUL_FILTERS
export QUANTEMU_PRECISION_MATMUL_GRADS
export QUANTEMU_PRECISION_MUL_INPUTS
export QUANTEMU_PRECISION_MUL_GRADS

export QUANTEMU_PRECISION_SIGMOID_INPUTS
export QUANTEMU_PRECISION_SIGMOID_GRADS
export QUANTEMU_PRECISION_TANH_INPUTS
export QUANTEMU_PRECISION_TANH_GRADS 

export QUANTEMU_ALLOCATE_COPY_INPUTS
export QUANTEMU_ALLOCATE_COPY_FILTERS
export QUANTEMU_ALLOCATE_COPY_GRADS
export QUANTEMU_CBLOCK_TYPE_CONV_INPUTS
export QUANTEMU_CBLOCK_TYPE_CONV_FILTERS
export QUANTEMU_CBLOCK_TYPE_CONV_GRADS
export QUANTEMU_CBLOCK_TYPE_BNORM_INPUTS
export QUANTEMU_CBLOCK_TYPE_BNORM_GRADS
export QUANTEMU_CBLOCK_SIZE_INPUTS
export QUANTEMU_CBLOCK_SIZE_FILTER
export QUANTEMU_CBLOCK_SIZE_GRAD
