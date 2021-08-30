(define (problem rovers-3-2-rover0) (:domain rovers-3-2-rover0)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mars - objective
    mothership - lander
    rover0RockStore - store
    rover0soilStore - store
    rover1RockStore - store
    rover1soilStore - store
    rover0cam - camera
    rover1cam - camera
    color - mode


	(:private
		rover0 - rover
	)
)
(:init

    ;; private
	(store_of rover0RockStore rover0)
	(store_of rover0soilStore rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_soil_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 base quarry)
	(can_traverse rover0 quarry base)
	(can_traverse rover0 base dune)
	(can_traverse rover0 dune base)
	(on_board rover0cam rover0)
	(at rover0 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(visible_from mars dune)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover0soilStore)
	(empty rover1RockStore)
	(empty rover1soilStore)
	(supports rover0cam color)
	(supports rover1cam color)
	(calibration_target rover0cam mars)
	(calibration_target rover1cam mars)

)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
		(communicated_image_data mars color)
	)
)
)