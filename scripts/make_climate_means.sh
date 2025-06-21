#!/bin/bash
set -e

SCALE_DIV=10.0
OUTDIR="worldclim.org"

# Averaged temperature: tavg, tmin, tmax
for VAR in tavg tmin tmax; do
  echo "📈 Processing $VAR..."
  gdal_calc.py \
    -A ${OUTDIR}/wc2.1_30s_${VAR}_01.tif -B ${OUTDIR}/wc2.1_30s_${VAR}_02.tif -C ${OUTDIR}/wc2.1_30s_${VAR}_03.tif \
    -D ${OUTDIR}/wc2.1_30s_${VAR}_04.tif -E ${OUTDIR}/wc2.1_30s_${VAR}_05.tif -F ${OUTDIR}/wc2.1_30s_${VAR}_06.tif \
    -G ${OUTDIR}/wc2.1_30s_${VAR}_07.tif -H ${OUTDIR}/wc2.1_30s_${VAR}_08.tif -I ${OUTDIR}/wc2.1_30s_${VAR}_09.tif \
    -J ${OUTDIR}/wc2.1_30s_${VAR}_10.tif -K ${OUTDIR}/wc2.1_30s_${VAR}_11.tif -L ${OUTDIR}/wc2.1_30s_${VAR}_12.tif \
    --calc="(A+B+C+D+E+F+G+H+I+J+K+L)/12.0/${SCALE_DIV}" \
    --NoDataValue=-9999 --overwrite \
    --outfile=${OUTDIR}/${VAR}_annual_mean.tif
done

# Sum of precipitation (mm)
echo "🌧️ Processing prec..."
gdal_calc.py \
  -A ${OUTDIR}/wc2.1_30s_prec_01.tif -B ${OUTDIR}/wc2.1_30s_prec_02.tif -C ${OUTDIR}/wc2.1_30s_prec_03.tif \
  -D ${OUTDIR}/wc2.1_30s_prec_04.tif -E ${OUTDIR}/wc2.1_30s_prec_05.tif -F ${OUTDIR}/wc2.1_30s_prec_06.tif \
  -G ${OUTDIR}/wc2.1_30s_prec_07.tif -H ${OUTDIR}/wc2.1_30s_prec_08.tif -I ${OUTDIR}/wc2.1_30s_prec_09.tif \
  -J ${OUTDIR}/wc2.1_30s_prec_10.tif -K ${OUTDIR}/wc2.1_30s_prec_11.tif -L ${OUTDIR}/wc2.1_30s_prec_12.tif \
  --calc="A+B+C+D+E+F+G+H+I+J+K+L" \
  --NoDataValue=-9999 --overwrite \
  --outfile=${OUTDIR}/prec_annual_sum.tif

# Vapor pressure and wind (mean, unscaled)
for VAR in vapr wind; do
  echo "🌬️ Processing $VAR..."
  gdal_calc.py \
    -A ${OUTDIR}/wc2.1_30s_${VAR}_01.tif -B ${OUTDIR}/wc2.1_30s_${VAR}_02.tif -C ${OUTDIR}/wc2.1_30s_${VAR}_03.tif \
    -D ${OUTDIR}/wc2.1_30s_${VAR}_04.tif -E ${OUTDIR}/wc2.1_30s_${VAR}_05.tif -F ${OUTDIR}/wc2.1_30s_${VAR}_06.tif \
    -G ${OUTDIR}/wc2.1_30s_${VAR}_07.tif -H ${OUTDIR}/wc2.1_30s_${VAR}_08.tif -I ${OUTDIR}/wc2.1_30s_${VAR}_09.tif \
    -J ${OUTDIR}/wc2.1_30s_${VAR}_10.tif -K ${OUTDIR}/wc2.1_30s_${VAR}_11.tif -L ${OUTDIR}/wc2.1_30s_${VAR}_12.tif \
    --calc="(A+B+C+D+E+F+G+H+I+J+K+L)/12.0" \
    --NoDataValue=-9999 --overwrite \
    --outfile=${OUTDIR}/${VAR}_annual_mean.tif
done

echo "✅ All climate aggregates complete."
